from easyDataverse import Dataverse

def load_dataset(dset_name, filedir="./data/", n_parallel_downloads=1):
    """
    Load BIDS-formatted datasets available through MUniverse

    Available datasets include:
    - Caillet et. al. 2023: "doi:10.7910/DVN/F9GWIW"
    - Avrillon et. al. 2024: "doi:10.7910/DVN/L9OQY7"
    - Grison et. al. 2025: "doi:10.7910/DVN/ID1WNQ"
    - MUniverse Neuromotion-Train set: "doi:10.7910/DVN/2UQHTP"
    - MUinverse Neuromotion-Test set: "doi:10.7910/DVN/QYI336"
    - MUniverse Hybrid-Tibialis set: "doi:10.7910/DVN/YHTGGA"

    """
    dataverse = Dataverse(server_url="https://dataverse.harvard.edu/")
    dataset = dataverse.load_dataset(
        pid=dset_name,
        filedir=filedir,
        n_parallel_downloads=n_parallel_downloads,
    )
    return dataset
