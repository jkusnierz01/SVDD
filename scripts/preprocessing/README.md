# Preprocessing pipeline

**Pełna dokumentacja:** [`docs/preprocessing_pipeline.md`](../../docs/preprocessing_pipeline.md)

## Szybki start

```bash
cp scripts/preprocessing/paths.env.example scripts/preprocessing/paths.env
# edytuj ścieżki scratch

chmod +x scripts/preprocessing/run.sh

# Sonics — clean (Stage 1 + 2)
./scripts/preprocessing/run.sh sonics features
./scripts/preprocessing/run.sh sonics pool

# OOD
./scripts/preprocessing/run.sh mom features && ./scripts/preprocessing/run.sh mom pool
./scripts/preprocessing/run.sh m6 features && ./scripts/preprocessing/run.sh m6 pool
```

## Etapy

| Stage | Komenda | Output |
|-------|---------|--------|
| 0 audio | `run.sh sonics audio` | `all_data_16k_mono/` |
| 1 features | `run.sh sonics features` | `preprocessed_16k_mono/` |
| 2 pool | `run.sh sonics pool` | `preprocessed_pooled_fp16/` |
| 1b aug | `run.sh sonics features-aug 0` | `preprocessed_aug_v0/` |
| 2b pool aug | `run.sh sonics pool-aug 0` | `preprocessed_pooled_aug_v0/` |
| aug-all | `run.sh sonics aug-all` | warianty 0–2 + merge |
| 3 merge | `run.sh sonics merge` | `preprocessed_pooled_merged/index.json` |
| trening merged | `experiments=bimamba_merged` | `data/sonics_merged.yaml` |

## Pliki

| Plik | Rola |
|------|------|
| `run.sh` | jedyny entrypoint |
| `paths.env.example` | ścieżki scratch |
| `../processing/precompute_pooled.py` | Stage 2 |
| `../processing/merge_pooled_index.py` | Stage 3 merge |
| `../../src/preprocess.py` | Stage 1 (Hydra) |

Stare skrypty w `scripts/bash/` nadal działają; `run.sh` je opakowuje lub wskazuje w dokumentacji.
