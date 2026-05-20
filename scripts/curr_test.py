import muniverse.datasets as md

config = {
    "input_config": "configs/neuromotion.json",
    "output_dir": "data/outputs/",
    "engine": "docker",
    "container": "pranavm19/muniverse:neuromotion",
    "cache_dir": "data/cache/",
}

md.simulate.validate_config(config.get("input_config"))
md.generate_recording(config)