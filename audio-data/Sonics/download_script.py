from huggingface_hub import snapshot_download

snapshot_download(repo_id="awsaf49/sonics", repo_type="dataset", local_dir="/mnt/data/kusnierz/audio-data/Sonics/downloads")