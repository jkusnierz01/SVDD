#!/bin/bash
#SBATCH -A plgdfsingingpwr-gpu
#SBATCH -p plgrid-gpu-v100
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --output=logs_resample_16k.txt

#script to resample all_data tracks to 16k (training needs it)

module load ffmpeg

DATA_DIR="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/all_data_32k"
OUT_DIR="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/all_data_16k_mp3"

mkdir -p "$OUT_DIR"

echo "Szukam plików FLAC do resampelowania..."
mapfile -d '' FILES < <(find "$DATA_DIR" -name "*.flac" -print0)
echo "Znaleziono: ${#FILES[@]} plików"

resample_file() {
    local src="$1"
    local dest="$OUT_DIR/$(basename "$src")"

    # pomiń jeśli już istnieje w docelowym folderze
    if [ -f "$dest" ]; then
        return 0
    fi

    ffmpeg -hide_banner -loglevel error -y -i "$src" -vn -ar 16000 -acodec flac "$dest"

    if [ $? -ne 0 ]; then
        rm -f "$dest"
        echo "BŁĄD: $src"
    fi
}

export OUT_DIR
export -f resample_file

echo "Rozpoczynam resampelowanie do 16kHz z $SLURM_CPUS_PER_TASK wątkami..."
printf '%s\0' "${FILES[@]}" | xargs -0 -P "$SLURM_CPUS_PER_TASK" -I{} bash -c 'resample_file "$@"' _ {}

echo "Gotowe."
