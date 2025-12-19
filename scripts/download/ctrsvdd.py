import os
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import gdown
import argparse


def download_file(url, dest_folder, filename=None):
    if filename is None:
        filename = os.path.basename(urlparse(url).path)
    dest_path = os.path.join(dest_folder, filename)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        desc=filename,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
    print(f"Downloaded {filename}")

def get_google_drive_download_url(view_url):
    parsed = urlparse(view_url)
    path_parts = parsed.path.split('/')
    if 'd' in path_parts:
        idx = path_parts.index('d')
        file_id = path_parts[idx + 1]
    else:
        raise ValueError("Invalid Google Drive URL")
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def get_dropbox_download_url(view_url):
    return view_url.replace('dl=0', 'dl=1')


def download_google_drive(file_id, dest_folder, filename):
    dest_path = os.path.join(dest_folder, filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    print(f"Downloaded {filename} from Google Drive")


def download_ctrsvdd(output_path: str):    
    zenodo_base = "https://zenodo.org/records/10467648/files/"
    zenodo_files = [
        "train_set.zip?download=1",
        "dev_set.zip?download=1",
        "dev.txt?download=1",
        "train.txt?download=1"
    ]
    dropbox_url = "https://www.dropbox.com/scl/fi/vnz3v19hzdvq085t9r0th/OFUTON_P_UTAGOE_DB.zip?rlkey=0qv5ewcoeqquejiefa582nd5a&e=1&dl=0"
    
    
    os.makedirs(output_path, exist_ok=True)
    for file in zenodo_files:
        url = zenodo_base + file
        download_file(url, output_path)
    
    try:
        download_google_drive("17VOlqPKT7ssnOTCp6_ZBIWht1-h8FAt7", output_path, "JVS_MuSiC.zip")
    except Exception as e:
        print(f"Błąd pobierania JVS_MuSiC.zip: {e}")
    
    try:
        download_google_drive("1a2BLDSVf4o8SdS019AxsZkJYaGprfdCx", output_path, "oniku_dataset.zip")
    except Exception as e:
        print(f"Błąd pobierania oniku_dataset.zip: {e}")
    
    download_url = get_dropbox_download_url(dropbox_url)
    try:
        download_file(download_url, output_path)
    except Exception as e:
        print(f"Błąd pobierania Ofuton: {e}")
        
        
    print("\n=== NEEDS TO BE DOWNLOADED ON THEIR OWN ===")
    print("1. Kiritan dataset: https://zunko.jp/kiridev/dl_voicezip.php")
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--download", action='store_true')

    args = parser.parse_args()
    output_path = args.output
    
    if args.download:
        download_ctrsvdd(output_path)
    

if __name__ == "__main__":
    main()