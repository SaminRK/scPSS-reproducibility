import gdown
import shutil
import os

"""
This data is originally from HLCA and it is sampled with cells from healthy and IPF individuals
in the MultiMIL paper: https://multimil.readthedocs.io/en/latest/notebooks/mil_classification.html
"""

def download_hlca_tutorial(out_path):
    """
    Downloads the HLCA tutorial h5ad file from Google Drive and moves it to out_path.
    
    Parameters:
    - out_path (str): Destination path where the file should be saved, including the filename (e.g. '/path/to/hcla_tutorial.h5ad')
    """
    url = "https://drive.google.com/uc?export=download&id=1wWGwbPeap-IqWNVlwVVUWVrUAMrf45ye"
    tmp_file = "hlca_tutorial.h5ad"

    # Download file to current working directory
    gdown.download(url, output=tmp_file, quiet=False)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Move to desired path
    shutil.move(tmp_file, out_path)
