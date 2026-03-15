#!/bin/bash
#SBATCH --job-name=mass_audio_laundry
#SBATCH --account=plgdfsingingpwr-cpu
#SBATCH --partition=plgrid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=logs_mass_laundering.txt
# Re-encodes all FLAC files through MP3 at a fixed bitrate and back to FLAC.

module load ffmpeg parallel

INPUT_DIR="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/all_data"
OUTPUT_DIR="/net/people/plgrid/plgjedrzejkusnierz/scratch/data/Sonics/all_data_32k"
BITRATE="32k"
JOBS="16"


if ! command -v ffmpeg &>/dev/null; then
    echo "ffmpeg not installed"
    exit 1
fi

if ! command -v parallel &>/dev/null; then
    echo "GNU parallel not installed"
    exit 1
fi

echo "Input:    $INPUT_DIR"
echo "Output:   $OUTPUT_DIR"
echo "Bitrate:  $BITRATE"
echo "Jobs:     $JOBS"
echo ""

mapfile -t FILES < <(find "$INPUT_DIR" -name "*.flac" -type f | shuf -n 2000)
TOTAL=${#FILES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "no files in: $INPUT_DIR"
    exit 1
fi

echo "$TOTAL FLAC files"
echo ""

process_audio() {
    in_file="$1"
    out_file="${in_file/$2/$3}"
    tmp_mp3="${out_file%.flac}.mp3"

    mkdir -p "$(dirname "$out_file")"

    if [[ ! -f "$out_file" ]]; then
        ffmpeg -y -hide_banner -loglevel error -i "$in_file" -b:a "$4" "$tmp_mp3" && \
        ffmpeg -y -hide_banner -loglevel error -i "$tmp_mp3" "$out_file"
        rm -f "$tmp_mp3"
    fi
}
export -f process_audio

echo "Starting..."
printf '%s\n' "${FILES[@]}" | parallel --jobs "$JOBS" --bar process_audio {} "$INPUT_DIR" "$OUTPUT_DIR" "$BITRATE"
echo "Done!"