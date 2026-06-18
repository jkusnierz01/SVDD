# Pipeline przygotowania danych (SVDD)

Jeden dokument opisujący **cały** przepływ od surowego audio do tensorów `[1500, 2048]` używanych w treningu BiMamba.

**Punkt wejścia (skrypty):** `scripts/preprocessing/run.sh`  
**Konfiguracja Hydra (ekstrakcja cech):** `src/configs/preprocess.yaml` + `src/configs/preprocessing/*.yaml`

---

## Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Audio na dysku (CPU, opcjonalnie)                                │
│  scripts/bash/sonics_mp3_codec.sh  →  all_data_16k_mono/                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Ekstrakcja W2V + MERT (GPU, src/preprocess.py)                   │
│  Wejście:  audio 16 kHz                                                     │
│  Wyjście:  {out}/wav2vec/*.npy [6000,1024]                                   │
│            {out}/mert/*.npy    [6000,1024]                                  │
│            {out}/index.json                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Pool + concat (CPU, scripts/processing/precompute_pooled.py)     │
│  Wyjście:  {out}/*.npy [1500,2048] fp16                                     │
│            {out}/index.json  (pole "pooled")                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
         ┌──────────────────────┐           ┌──────────────────────┐
         │  STAGE 3a — Trening  │           │  STAGE 1b + 2b (opt) │
         │  clean index         │           │  audio aug, train    │
         │  → train.py          │           │  only → merge index  │
         └──────────────────────┘           └──────────────────────┘
```

---

## Etapy — co, gdzie, jak

### Stage 0 — Przygotowanie audio (Sonics)

| | |
|---|---|
| **Skrypt** | `scripts/bash/sonics_mp3_codec.sh` |
| **Wejście** | `scratch/data/Sonics/all_data/` |
| **Wyjście** | `scratch/data/Sonics/all_data_16k_mono/` |
| **Kiedy** | Raz na początku; nie powtarzać bez powodu |

Inne datasety (SingFake): `scripts/processing/simulate_codec_singfake.py`, `split_singfake.py`, itd.

---

### Stage 1 — Ekstrakcja embeddingów (GPU)

| | |
|---|---|
| **Skrypt** | `python src/preprocess.py` (+ config datasetu) |
| **Kod** | `src/preprocessing/base.py`, `src/preprocessing/sonics.py` |
| **Config** | `src/configs/preprocess.yaml` → `paths.output_dir` |
| | `src/configs/preprocessing/sonics.yaml` (Sonics) |
| | `src/configs/preprocessing/mom.yaml`, `m6.yaml` (OOD) |

**Logika:** 120 s → 4×30 s chunki → każdy chunk osobno W2V @16 kHz + MERT @24 kHz → sklejenie w czasie → `[6000, 1024]` per modalność.

```bash
# Uruchamiaj z katalogu głównego repo. Stage 1 woła preprocess.py z src/ (Hydra configs).

# Sonics (clean)
./scripts/preprocessing/run.sh sonics features

# MoM / M6 (OOD eval)
./scripts/preprocessing/run.sh mom features
./scripts/preprocessing/run.sh m6 features
```

**Typowe katalogi wyjściowe:**

| Dataset | Stage 1 output |
|---------|----------------|
| Sonics clean | `.../Sonics/preprocessed_16k_mono/` |
| Sonics aug (plan) | `.../Sonics/preprocessed_aug_v{N}/` |
| MoM | `.../MoM/mom_preprocessed/` |
| M6 | `.../MoM/m6_preprocessed/` (ścieżka w configu) |

---

### Stage 2 — Precompute pooled (CPU)

| | |
|---|---|
| **Skrypt** | `scripts/processing/precompute_pooled.py` |
| **Wrapper SLURM** | `scripts/bash/precompute_pooled.sh` (edytuj INDEX/OUTPUT) |
| **Wejście** | `index.json` ze Stage 1 (pola `wav2vec`, `mert`) |
| **Wyjście** | `{stem}.npy` → `[1500, 2048]` fp16, `index.json` z polem `pooled` |

```bash
./scripts/preprocessing/run.sh sonics pool
```

**Typowe katalogi:**

| Dataset | Stage 2 output |
|---------|----------------|
| Sonics clean | `.../Sonics/preprocessed_pooled_fp16/` |
| Sonics aug | `.../Sonics/preprocessed_pooled_aug_v{N}/` |

---

### Stage 3 — Trening / eval

| | |
|---|---|
| **Config treningu** | `src/configs/data/sonics.yaml` → `data_dir` wskazuje na **Stage 2** |
| **Loader** | `src/data/sonics_dataloader.py` → `PooledFeaturesDataset` |
| **Augmentacje embeddingów** | `src/data/data_augmentation.py` (tylko w treningu, co epokę) |

```yaml
# src/configs/data/sonics.yaml
data_dir: .../preprocessed_pooled_fp16          # clean baseline
# data_dir: .../preprocessed_pooled_merged     # clean train + aug train
```

---

## Gałąź audio augmentacji (offline) — zaimplementowane

1. **Nie nadpisuj** `preprocessed_pooled_fp16/` (clean baseline).
2. Stage 1b: `./scripts/preprocessing/run.sh sonics features-aug 0` (train only, `sonics_aug.yaml`)
3. Stage 2b: `./scripts/preprocessing/run.sh sonics pool-aug 0`
4. Powtórz dla wariantów 1, 2 **albo** `./scripts/preprocessing/run.sh sonics aug-all`
5. Merge: `./scripts/preprocessing/run.sh sonics merge`
6. Trening: `python src/train.py experiments=bimamba_merged`

Augmentacje audio (gain, szum, krótki dropout, flip polaryzacji) — `src/preprocessing/audio_augmentation.py`, deterministyczne per `(stem, variant)`.

**Naming stemów aug:** `train_{id}_aug0_spoof` (etykieta nadal ostatnim segmentem).

**Valid / test / OOD:** zawsze tylko **clean** index.

---

## Mapa plików w repo

```
scripts/
├── preprocessing/
│   ├── run.sh              ← JEDEN entrypoint (stage 0–3, merge)
│   └── paths.env.example   ← ścieżki scratch (kopiuj → paths.env)
├── processing/
│   ├── precompute_pooled.py
│   └── merge_pooled_index.py
└── bash/
    ├── sonics_mp3_codec.sh      # stage 0
    ├── precompute_pooled.sh     # przykład SLURM stage 2
    └── ...

src/
├── preprocess.py               # stage 1 (Hydra)
├── preprocessing/
│   ├── base.py                 # ChunkedAudioDataset, BaseFeatureProcessor
│   └── sonics.py
└── configs/
    ├── preprocess.yaml
    └── preprocessing/
        ├── sonics.yaml
        ├── sonics_aug.yaml     # szablon gałęzi aug (output_dir aug)
        ├── mom.yaml
        └── m6.yaml
```

---

## Zasady

1. **Clean zostaje clean** — nowe eksperymenty dokładają katalogi, nie nadpisują.
2. **`index.json` to kontrakt** — loader czyta wyłącznie index + ścieżki w polach `pooled` / `wav2vec` / `mert`.
3. **Audio aug przed W2V/MERT** — nie da się sensownie augować już pooled `.npy`.
4. **Embedding aug w treningu** — osobna warstwa (`data_augmentation.py`), zostaje przy merged i clean.
5. **OOD eval** — MoM/M6 zawsze z clean preprocessingu.

---

## Szybka ściąga

```bash
cd /path/to/SVDD
cp scripts/preprocessing/paths.env.example scripts/preprocessing/paths.env
# edytuj paths.env

# Pełny Sonics clean (po stage 0):
./scripts/preprocessing/run.sh sonics features
./scripts/preprocessing/run.sh sonics pool

# Trening (bez zmian, z katalogu src lub root — jak dotąd):
python src/train.py experiments=bimamba

# Później — aug (3 warianty) + merge + trening:
# sbatch scripts/bash/sonics_preprocess_aug.sh 0   # GPU — powtórz 1, 2
# sbatch scripts/bash/precompute_aug_pooled.sh 0   # CPU — powtórz 1, 2
# ./scripts/preprocessing/run.sh sonics merge
# python src/train.py experiments=bimamba_merged
```
