import os
import requests
from urllib.parse import urlparse
import gdown
from tqdm import tqdm

# files
'''
https://zenodo.org/records/10467648/files/train_set.zip?download=1
https://zenodo.org/records/10467648/files/dev_set.zip?download=1
https://zenodo.org/records/10467648/files/dev.txt?download=1
https://zenodo.org/records/10467648/files/train.txt?download=1

Pozostałe datasety:
https://drive.google.com/file/d/17VOlqPKT7ssnOTCp6_ZBIWht1-h8FAt7/view?usp=sharing,
https://www.dropbox.com/scl/fi/vnz3v19hzdvq085t9r0th/OFUTON_P_UTAGOE_DB.zip?rlkey=0qv5ewcoeqquejiefa582nd5a&e=1&dl=0
https://zunko.jp/kiridev/dl_voicezip.php
https://drive.google.com/open?id=1a2BLDSVf4o8SdS019AxsZkJYaGprfdCx
'''


# Function to download a file from a URL
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

# Function to get Google Drive download URL
def get_google_drive_download_url(view_url):
    parsed = urlparse(view_url)
    path_parts = parsed.path.split('/')
    if 'd' in path_parts:
        idx = path_parts.index('d')
        file_id = path_parts[idx + 1]
    else:
        raise ValueError("Invalid Google Drive URL")
    return f"https://drive.google.com/uc?export=download&id={file_id}"

# Function to get Dropbox download URL
def get_dropbox_download_url(view_url):
    return view_url.replace('dl=0', 'dl=1')

# Function to download from Google Drive using gdown
def download_google_drive(file_id, dest_folder, filename):
    dest_path = os.path.join(dest_folder, filename)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
    print(f"Downloaded {filename} from Google Drive")

# Main download function
def main():
    dest_folder = 'downloads'
    os.makedirs(dest_folder, exist_ok=True)

    # Zenodo files
    zenodo_base = "https://zenodo.org/records/10467648/files/"
    zenodo_files = [
        "train_set.zip?download=1",
        "dev_set.zip?download=1",
        "dev.txt?download=1",
        "train.txt?download=1"
    ]
    for file in zenodo_files:
        url = zenodo_base + file
        download_file(url, dest_folder)

    # Google Drive files using gdown (obsługuje duże pliki i warningi)
    print("\nPobieranie z Google Drive (może wymagać potwierdzenia dla dużych plików):")
    try:
        download_google_drive("17VOlqPKT7ssnOTCp6_ZBIWht1-h8FAt7", dest_folder, "JVS_MuSiC.zip")
    except Exception as e:
        print(f"Błąd pobierania JVS_MuSiC.zip: {e}")
    
    try:
        download_google_drive("1a2BLDSVf4o8SdS019AxsZkJYaGprfdCx", dest_folder, "oniku_dataset.zip")
    except Exception as e:
        print(f"Błąd pobierania oniku_dataset.zip: {e}")

    # Dropbox file (Ofuton)
    dropbox_url = "https://www.dropbox.com/scl/fi/vnz3v19hzdvq085t9r0th/OFUTON_P_UTAGOE_DB.zip?rlkey=0qv5ewcoeqquejiefa582nd5a&e=1&dl=0"
    download_url = get_dropbox_download_url(dropbox_url)
    try:
        download_file(download_url, dest_folder)
    except Exception as e:
        print(f"Błąd pobierania Ofuton: {e}")

    # Linki wymagające ręcznego pobierania (z logowaniem/formularzem)
    print("\n=== UWAGA: Następujące pliki wymagają RĘCZNEGO pobrania ===")
    print("1. Kiritan dataset: https://zunko.jp/kiridev/dl_voicezip.php")
    print("   (Wymagane: logowanie + akceptacja licencji)")
    print("\nPo pobraniu skopiuj pliki do folderu 'downloads'.")

if __name__ == "__main__":
    main()
