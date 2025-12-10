# ae_decomposer.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse your existing utilities (same package structure as your CBSS snippet)
from .core import (
    bandpass_signals,
    notch_signals,
    extension,
    whitening,
    est_spike_times,
    remove_duplicates,
    remove_bad_sources,
    peel_off,
)

# ----------------------------
#   Building blocks
# ----------------------------

class OrthogonalEncoderSO(nn.Module):
    """
    Orthogonally constrained linear map V \in SO(Din), then we take the
    first Dout rows to get a rectangular encoder W = V[:Dout, :].

    Forward: y = W @ x, where x: [B, Din] (time-batch), y: [B, Dout].
    """
    def __init__(self, din: int, dout: int, init_scale: float = 1e-3, device="cpu", dtype=torch.float32):
        super().__init__()
        assert dout <= din, "dout must be <= din (take first rows of SO(din))."
        self.din = din
        self.dout = dout
        self.register_parameter(
            "A",
            nn.Parameter(init_scale * torch.randn(din, din, device=device, dtype=dtype))
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Skew-symmetric parameter -> Lie algebra so(n)
        # K = A - A^T, then V = expm(K) \in SO(n)
        K = self.A - self.A.transpose(-1, -2)
        V = torch.matrix_exp(K)  # [din, din], guaranteed orthogonal with det=+1 if initialized near 0
        W = V[: self.dout, :]    # take first rows -> rectangular
        y = x @ W.t()            # x: [B, din] -> y: [B, dout]
        return y, W  # return W so we can export filters later


class TanhshrinkLayer(nn.Module):
    def forward(self, x):
        return x - torch.tanh(x)


class EMGAutoencoder(nn.Module):
    """
    Encoder: y = ReLU( W x ), with W orthogonal rows from SO(Din)
    Decoder: x_hat = Tanhshrink( y @ Z^T + b ), fully unconstrained
    """
    def __init__(self, din: int, dlat: int, device="cpu", dtype=torch.float32):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.encoder = OrthogonalEncoderSO(din, dlat, device=device, dtype=dtype)
        self.relu = nn.ReLU(inplace=False)
        self.decoder = nn.Linear(dlat, din, bias=True, device=device, dtype=dtype)
        self.tanhshrink = TanhshrinkLayer()

        # Xavier init for decoder
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        """
        x: [B, Din]
        returns: x_hat [B, Din], s_hat [B, dlat], W [dlat, Din]
        """
        y_lin, W = self.encoder(x)      # [B, dlat], [dlat, Din]
        s_hat = self.relu(y_lin)        # nonnegative sparse spikes
        x_lin = self.decoder(s_hat)     # [B, Din]
        x_hat = self.tanhshrink(x_lin)  # denoise tiny values
        return x_hat, s_hat, W


# ----------------------------
#   Losses
# ----------------------------

def lp_lq_sparsity(s_hat: torch.Tensor, p=0.9, q=1.8, eps=1e-8):
    """
    Temporal sparsity penalty: log10( (q/p) * ||s||_p / ||s||_q ), averaged over units.
    s_hat: [B, dlat]
    We compute norms over time within a window; aggregate across batch as mean.
    """
    # Sum across batch -> a time-aggregated view for this mini-batch
    # (treat batch as temporal slices)
    ss = torch.clamp(s_hat, min=0.0)  # already ReLU but be safe
    # Compute norms per-unit across batch
    # Avoid zero by eps to keep gradient stable
    lp = torch.pow(torch.sum(torch.pow(ss + eps, p), dim=0) + eps, 1.0 / p)   # [dlat]
    lq = torch.pow(torch.sum(torch.pow(ss + eps, q), dim=0) + eps, 1.0 / q)   # [dlat]
    ratio = (q / p) * (lp / (lq + eps))
    pen = torch.log10(ratio + eps).mean()
    return pen


# ----------------------------
#   Main API (mirrors CBSS)
# ----------------------------

class AEDecoderConfig:
    def __init__(self, **kwargs):
        # Preprocessing
        self.bandpass = [20, 500]
        self.bandpass_order = 2
        self.notch_frequency = 50
        self.notch_n_harmonics = 3
        self.notch_order = 2
        self.notch_width = 1

        # Temporal extension
        self.ext_fact = 2  # R (paper found R=2 best)
        # Whitening
        self.whitening_method = "ZCA"
        self.whitening_reg = "auto"

        # Model dims
        self.latent_dim = None  # default -> number of original channels m

        # Training
        self.epochs = 100
        self.batch_size = 5000         # samples per batch (time-slices)
        self.learning_rate = 8e-3
        self.sparsity_p = 0.9
        self.sparsity_q = 1.8
        self.lambda_sparsity = 1.0
        self.weight_decay = 0.0
        self.device = "cpu"
        self.dtype = torch.float32
        self.random_seed = 1909
        self.shuffle_windows = True

        # Post-processing / evaluation
        self.cluster_method = "kmeans"
        self.sil_th = 0.9
        self.cov_th = 0.30
        self.min_num_spikes = 10
        self.match_th = 0.3
        self.match_max_shift = 0.1
        self.match_tol = 0.001

        # Optional iterative peeling (like CBSS)
        self.enable_peel_off = True
        self.peel_win = 0.025

        for k, v in kwargs.items():
            setattr(self, k, v)


class AEDecoder:
    """
    Autoencoder-based EMG decomposition (unsupervised), closely following Mayer et al. (2023).
    Public API mirrors CBSS.decompose(sig, fsamp).
    """
    def __init__(self, config: AEDecoderConfig | None = None, **kwargs):
        self.cfg = config if config is not None else AEDecoderConfig()
        for k, v in kwargs.items():
            setattr(self.cfg, k, v)

        # For reproducibility
        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)

    def _to_torch(self, x: np.ndarray) -> torch.Tensor:
        # Convert device and dtype to proper PyTorch objects
        device = torch.device(self.cfg.device) if isinstance(self.cfg.device, str) else self.cfg.device
        if isinstance(self.cfg.dtype, str):
            dtype = getattr(torch, self.cfg.dtype)
        else:
            dtype = self.cfg.dtype
        return torch.from_numpy(x).to(device=device, dtype=dtype)

    def _prep_signal(self, sig: np.ndarray, fsamp: float):
        """
        Preprocess:
          - bandpass / notch
          - temporal extension
          - mean removal & edge zeroing like CBSS
          - whitening (returns whitened signal and whitening matrix Z)
        """
        # Bandpass
        if self.cfg.bandpass is not None:
            sig = bandpass_signals(
                sig, fsamp,
                high_pass=self.cfg.bandpass[0],
                low_pass=self.cfg.bandpass[1],
                order=self.cfg.bandpass_order
            )
        # Notch
        if self.cfg.notch_frequency is not None:
            sig = notch_signals(
                sig, fsamp,
                nfreq=self.cfg.notch_frequency,
                dfreq=self.cfg.notch_width,
                order=self.cfg.notch_order,
                n_harmonics=self.cfg.notch_n_harmonics,
            )

        # Temporal extension
        ext_sig = extension(sig, self.cfg.ext_fact)  # [m*R, T]
        ext_sig = ext_sig - np.mean(ext_sig, axis=1, keepdims=True)

        # edge zeroing (like CBSS)
        cut = self.cfg.ext_fact * 2
        ext_sig[:, :cut] = 0
        ext_sig[:, -cut:] = 0

        # Whitening
        white_sig, Z = whitening(
            Y=ext_sig,
            method=self.cfg.whitening_method,
            backend="svd",
            regularization=self.cfg.whitening_reg,
        )
        return white_sig, Z  # shapes: [mR, T], [mR, mR]

    def _iter_minibatches(self, Xw: np.ndarray):
        """
        Yield batched [B, Din] slices from Xw.T (Din = mR).
        """
        T = Xw.shape[1]
        Din = Xw.shape[0]
        idx = np.arange(T)
        if self.cfg.shuffle_windows:
            np.random.shuffle(idx)
        # We construct batches by selecting *time indices* (treat time as batch)
        for start in range(0, T, self.cfg.batch_size):
            sel = idx[start : start + self.cfg.batch_size]
            xb = Xw[:, sel].T  # [B, Din]
            yield xb

    def _build_model(self, din: int, m_orig: int):
        dlat = self.cfg.latent_dim if self.cfg.latent_dim is not None else m_orig
        model = EMGAutoencoder(
            din=din,
            dlat=dlat,
            device=self.cfg.device,
            dtype=self.cfg.dtype
        )
        # Convert device to proper PyTorch object
        device = torch.device(self.cfg.device) if isinstance(self.cfg.device, str) else self.cfg.device
        return model.to(device)

    def _train(self, model: EMGAutoencoder, Xw: np.ndarray):
        optim = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay
        )

        Din = Xw.shape[0]
        model.train()
        for epoch in range(self.cfg.epochs):
            epoch_loss = 0.0
            nb = 0
            for xb_np in self._iter_minibatches(Xw):
                xb = self._to_torch(xb_np)  # [B, Din]
                optim.zero_grad()
                x_hat, s_hat, _ = model(xb)
                # Reconstruction on whitened, extended space (like paper)
                rec = F.mse_loss(x_hat, xb)
                # Temporal sparsity
                sp = lp_lq_sparsity(s_hat, p=self.cfg.sparsity_p, q=self.cfg.sparsity_q)
                loss = rec + self.cfg.lambda_sparsity * sp
                loss.backward()
                optim.step()

                epoch_loss += loss.item()
                nb += 1
            # (Optional) print training progress
            # print(f"[AE] epoch {epoch+1}/{self.cfg.epochs}  loss={epoch_loss/max(1,nb):.5f}")
        return model

    def _infer_sources_full(self, model: EMGAutoencoder, Xw: np.ndarray):
        """
        Run the full sequence through the encoder to get latent sources (over time).
        Returns:
          S: [dlat, T]  (nonnegative spikes)
          W: [dlat, Din] (encoder filters over whitened-extended space)
        """
        model.eval()
        with torch.no_grad():
            Xw_t = self._to_torch(Xw.T)  # [T, Din]
            x_hat, s_hat, W = model(Xw_t)  # s_hat: [T, dlat]
            S = s_hat.transpose(0, 1).cpu().numpy()  # [dlat, T]
            W_np = W.detach().cpu().numpy()          # [dlat, Din]
        return S, W_np

    def _postprocess(self, S: np.ndarray, fsamp: float, Xw: np.ndarray):
        """
        Spike picking, duplicates, bad sources. Also (optionally) peel components and refit
        to reduce interference.
        """
        n_mu, T = S.shape
        spikes = {i: [] for i in range(n_mu)}
        sil = np.zeros(n_mu)

        # spike picking on each latent
        for i in range(n_mu):
            spk, si = est_spike_times(
                S[i, :], fsamp, cluster=self.cfg.cluster_method
            )
            spikes[i] = spk
            sil[i] = si

        # Optional peel-off in whitened domain (helps reduce overlap influence)
        if self.cfg.enable_peel_off:
            Xw_res = Xw.copy()
            for i in range(n_mu):
                if len(spikes[i]) == 0:
                    continue
                # peel_off expects channels x time; Xw_res is [Din, T] already
                Xw_res, _, _ = peel_off(
                    Xw_res, spikes[i],
                    win=self.cfg.peel_win,
                    fsamp=fsamp
                )

        # Remove duplicates
        sources, spikes, sil, _ = remove_duplicates(
            S, spikes, sil, np.zeros((Xw.shape[0], n_mu)), fsamp,
            max_shift=self.cfg.match_max_shift,
            tol=self.cfg.match_tol,
            threshold=self.cfg.match_th
        )
        # Remove bad
        sources, spikes, sil, _ = remove_bad_sources(
            sources, spikes, sil, np.zeros((Xw.shape[0], sources.shape[0])),
            threshold=self.cfg.sil_th, min_num_spikes=self.cfg.min_num_spikes
        )
        return sources, spikes, sil

    def decompose(self, sig: np.ndarray, fsamp: float):
        """
        Args:
            sig: EMG (n_channels x n_samples)
            fsamp: sampling frequency (Hz)

        Returns:
            sources (ndarray): Estimated spike responses (n_mu x n_samples) in latent space
            spikes (dict): Sample indices of motor neuron discharges
            sil (ndarray): Silhouette-like scores per source
            mu_filters (ndarray): Encoder filters over whitened-extended space [dlat, mR]
        """
        # ---- Preprocess (extension + whitening) ----
        Xw, Z = self._prep_signal(sig, fsamp)  # [mR, T], [mR, mR]
        mR, T = Xw.shape
        m = sig.shape[0]
        Din = mR

        # ---- Build & train AE ----
        model = self._build_model(din=Din, m_orig=m)
        model = self._train(model, Xw)

        # ---- Infer latent sources across full sequence ----
        S, W = self._infer_sources_full(model, Xw)  # S: [dlat, T], W: [dlat, Din]

        # ---- Postprocess (spikes, duplicates, pruning) ----
        sources, spikes, sil = self._postprocess(S, fsamp, Xw)

        # NOTE about mu_filters:
        #   In the paper, sources are obtained as s = V * xw (after whitening & extension).
        #   Our W returned is exactly the encoder rows applied on whitened-extended inputs.
        #   If you want filters in the *pre-whitened extended* domain, multiply by Z^{-1}.
        #   Here we return W (whitened domain) to be consistent with forward usage.
        mu_filters = W  # [dlat, mR] rows are decomposition filters over whitened-extended space

        # Match output signature of CBSS
        # sources must be (n_mu x n_samples). Our latent 'sources' are over [T]; we already have that.
        return sources, spikes, sil, mu_filters
