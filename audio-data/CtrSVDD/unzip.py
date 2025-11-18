import zipfile
import os

input_dir = "/mnt/data/kusnierz/audio-data/CtrSVDD/downloads"  
output_dir = "/mnt/data/kusnierz/audio-data/CtrSVDD/downloads"  

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if fname.endswith(".zip"):
        zip_path = os.path.join(input_dir, fname)
        print(f"Wypakowuję: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
print("Gotowe!")