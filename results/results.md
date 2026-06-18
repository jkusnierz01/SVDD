# Wyniki eksperymentów BiMamba — detekcja deepfake muzyki

Dokument zbiera wyniki ewaluacji modeli trenowanych **wyłącznie na Sonics** (in-distribution) i testowanych out-of-distribution na **MoM** oraz **M6**. Niższy **EER** oznacza lepszy wynik.

---

## 1. Kontekst eksperymentów

| Aspekt | Opis |
|--------|------|
| **Zadanie** | Klasyfikacja binarna: utwór autentyczny vs. syntetyczny/deepfake |
| **Trening** | Zbiór Sonics (`PooledFeaturesDataset`, augmentacje włączone) |
| **Ewaluacja OOD** | MoM, M6 — bez augmentacji (`transform: null`) |
| **Metryka główna** | **EER** (Equal Error Rate) — próg, przy którym FAR = FRR |
| **Metryki pomocnicze** | F1, precision, recall (przy progu optymalnym dla F1) |
| **Embeddingi** | Wav2Vec2-XLS-R-300M (1024-d) + MERT-v1-330M (1024-d) |
| **Wejście modelu** | Tensor `[B, 1500, 2048]` — po precompute: concat + `AvgPool1d(4,4)` |

Wszystkie checkpointy trenowane na Sonics osiągają ~100% accuracy in-distribution; **problem badawczy to generalizacja OOD**, nie dopasowanie do Sonics.

---

## 2. Pipeline danych

```
120 s utworu → chunki 30 s → Wav2Vec + MERT (6000 klatek)
→ wyrównanie osi czasu → concat [6000, 2048]
→ AvgPool1d(4,4) → zapis fp16 [1500, 2048] na dysk
```

Precompute: `scripts/processing/precompute_pooled.py`. Model **nie** wykonuje poolingu — operuje na gotowych sekwencjach 1500 kroków.

Layout cech w wymiarze 2048: `[..., :1024]` = Wav2Vec, `[..., 1024:]` = MERT.

---

## 3. Architektura modeli

### 3.1 Early fusion — `BidirectionalMambaModel`

```
[W2V | MERT] → (norm?) → fuzja (concat / GMU) → BiMamba (fwd + bwd)
→ mean po czasie → Linear → logit
```

- **Concat (domyślnie):** `[B, T, 2048]` trafia do wspólnego enkodera BiMamba.
- **GMU per-feature (`fusion_type: gmu`):** bramka na każdy wymiar fuzji (`GMU_per_feature` w `fusion.py`) — najwięcej parametrów, fuzja przed enkoderem.
- **Scalar GMU (`fusion_type: scalar-gmu`):** jedna bramka na każdy krok czasu — mniej parametrów niż GMU per-feature.

Klasyfikator: `Linear(2 × d_model → 1)` po uśrednieniu sekwencji.

### 3.2 Late fusion — `LateFusionBidirectionalMambaModel`

```
W2V → BiMamba_w2v ─┐
                    ├─ concat [fwd‖bwd] per modality → concat(W2V, MERT) → mean → Linear
MERT → BiMamba_mert ┘
```

Każda modalność ma **osobny** stos Mamba (forward + backward). Fuzja następuje **po** enkodowaniu temporalnym. Przy `fusion_type: concat` wymiar przed klasyfikatorem to `4 × d_model`.

### 3.3 Loss

`SmoothedBCEWithLogitsLoss` z `label_smoothing=0.1` (domyślnie) — przy idealnym dopasowaniu loss ≈ **0.1985**. Przy `label_smoothing=0.0` używany jest zwykły `BCEWithLogitsLoss`.

---

## 4. Augmentacje (dataloader) vs. normalizacja (model)

### Augmentacje — `EmbeddingDataAugmentation` (tylko trening Sonics)

| Augmentacja | Parametry | Efekt |
|-------------|-----------|-------|
| **SpecAugment** | p=0.5 | Zeruje prostokątny blok (losowy czas × losowe wymiary cech) |
| **TemporalDropout** | p=0.3, max 20% sekwencji | Zeruje ciągły fragment osi czasu — model nie może polegać na jednym fragmencie |
| **FeatureDimDropout** | p=0.3, max 15% wymiarów | Zeruje losowe wymiary we wszystkich krokach czasu |
| **LatentGaussianNoise** | p=0.5, σ=0.05 | Szum Gaussa na całym tensorze |

### Normalizacja — `norm_type` w forward modelu (nie w dataloaderze)

| Wartość | Opis |
|---------|------|
| `global` | InstanceNorm na pełnym tensorze po fuzji concat |
| `per_modality` | Osobny InstanceNorm na W2V i MERT przed fuzją |
| `none` | Brak normalizacji |

---

## 5. Wspólne ustawienia serii eksperymentów

### Early fusion (v1–v8, oprócz wyjątków poniżej)

| Parametr | Wartość (v2–v8) | v9 (mniejszy model) |
|----------|-----------------|---------------------|
| Klasa | `BidirectionalMambaModel` | j.w. |
| `d_model` | 512 | **256** |
| `n_layers` | 4 | **2** |
| `dropout` | 0.3 | **0.5** |
| `label_smoothing` | 0.1 (v8: **0.0**) | **0.1** |
| `fusion_type` | concat (v6: gmu, v7: scalar-gmu) | concat |
| `fusion_dim` | 1024 (GMU) | 2048 |
| Optimizer | AdamW, lr=1e-4, wd=0.01 | wd=**0.05** |

### Late fusion (`late-fusion-bimamba-v1`)

| Parametr | Wartość |
|----------|---------|
| Klasa | `LateFusionBidirectionalMambaModel` |
| `d_model` | 512 |
| `n_layers` | 2 (×4 niezależne stosy Mamba: W2V fwd/bwd, MERT fwd/bwd) |
| `dropout` | 0.3 |
| `label_smoothing` | 0.1 |
| `fusion_type` | concat (late) |
| `norm_type` | none |

---

## 6. Tabela zbiorcza wyników OOD

| # | Eksperyment | Zmiana względem poprzedniego kroku | MoM EER ↓ | M6 EER ↓ | MoM F1 | M6 F1 |
|---|-------------|-------------------------------------|-----------|----------|--------|-------|
| 1 | **v1** | Mixup + bez TemporalDropout i InstanceNorm | 0.194 | 0.265 | 0.730 | 0.699 |
| 2 | **v2** (baseline) | Pełne augmentacje + `norm_type: global` | 0.344 | 0.364 | 0.680 | 0.688 |
| 3 | **v3** | Wyłączenie TemporalDropout | 0.534 | 0.590 | 0.432 | 0.187 |
| 4 | **v4** | Wyłączenie InstanceNorm (`norm_type: none`) | **0.158** | 0.344 | 0.836 | 0.747 |
| 5 | **v5** | `norm_type: per_modality` | 0.172 | 0.364 | 0.831 | 0.711 |
| 6 | **v6** | Early GMU per-feature (`fusion_type: gmu`) | 0.457 | 0.324 | 0.671 | 0.740 |
| 7 | **v7** | Early Scalar GMU (`fusion_type: scalar-gmu`) | 0.169 | 0.437 | 0.823 | 0.676 |
| 8 | **late-fusion v1** | Osobne enkodery BiMamba + late concat | **0.142** | **0.238** | 0.821 | 0.726 |
| 9 | **v8** | `label_smoothing: 0.0` | 0.192 | 0.426 | 0.752 | 0.702 |
| 10 | **v9** | Mniejszy model (256-d, 2 warstwy) | 0.261 | 0.511 | 0.720 | 0.693 |

---

## 7. Szczegóły poszczególnych eksperymentów

### v1 — mixup, bez TemporalDropout i InstanceNorm

**Model:** `BidirectionalMambaModel`, early fusion, `fusion_type: concat`.

**Konfiguracja względem późniejszej serii:**
- **Mixup** — mieszanie par próbek w batchu (interpolacja wejść i etykiet), dodatkowa regularizacja poza standardowymi augmentacjami embeddingów.
- **Bez TemporalDropout** — wyłączone maskowanie ciągłego fragmentu osi czasu (patrz sekcja 4).
- **Bez InstanceNorm** — `norm_type: none`; brak normalizacji w forward modelu.
- Pozostałe augmentacje embeddingów (SpecAugment, FeatureDimDropout, LatentGaussianNoise) włączone.

**Concat:** W2V `[B, T, 1024]` i MERT `[B, T, 1024]` są łączone w `[B, T, 2048]` i wspólnie trafiają do jednego enkodera BiMamba.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.194 | 0.265 |
| F1 | 0.730 | 0.699 |
| Precision | 0.586 | 0.537 |
| Recall | 0.970 | 1.000 |

---

### v2 — baseline

**Model:** `BidirectionalMambaModel`, early fusion, `fusion_type: concat`.

**Konfiguracja:**
- Pełny zestaw augmentacji embeddingów (SpecAugment, TemporalDropout, FeatureDimDropout, LatentGaussianNoise).
- **`norm_type: global`** — po concat W2V i MERT stosowany jest InstanceNorm1d na całym tensorze `[B, T, 2048]`. Normalizuje każdy wymiar cech osobno względem statystyk danej próbki (średnia i wariancja po osi czasu).
- `d_model=512`, `n_layers=4`, `dropout=0.3`, `label_smoothing=0.1`.

**Concat:** jak w v1 — obie modalności łączone przed wspólnym enkoderem temporalnym.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.344 | 0.364 |
| F1 | 0.680 | 0.688 |
| Precision | 0.589 | 0.604 |
| Recall | 0.806 | 0.800 |

---

### v3 — bez TemporalDropout

**Model:** jak v2 (early fusion, concat, `norm_type: global`).

**Zmiana:** wyłączony **TemporalDropout** — model nie dostaje losowo wyzerowanych ciągłych odcinków sekwencji (do ~20% długości). SpecAugment, FeatureDimDropout i LatentGaussianNoise pozostają aktywne.

**Cel ablacji:** sprawdzenie wpływu wymuszania niezależności od konkretnego fragmentu 30 s utworu.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.534 | 0.590 |
| F1 | 0.432 | 0.187 |
| Precision | 0.521 | 0.370 |
| Recall | 0.370 | 0.125 |

---

### v4 — bez InstanceNorm (`norm_type: none`)

**Model:** jak v2/v3, ale **`norm_type: none`** — brak InstanceNorm w forward (ani globalnego, ani per-modalność).

**Konfiguracja:** pełne augmentacje embeddingów, w tym TemporalDropout. Embeddingi W2V i MERT trafiają do concat w oryginalnej skali z precompute.

**Concat:** W2V i MERT łączone w `[B, T, 2048]` → wspólny BiMamba → mean po czasie → klasyfikator.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.158 | 0.344 |
| F1 | 0.836 | 0.747 |
| Precision | 0.765 | 0.627 |
| Recall | 0.921 | 0.925 |

---

### v5 — normalizacja per-modalność

**Model:** early fusion, concat.

**Zmiana względem v4:** **`norm_type: per_modality`** — przed concat osobno normalizowany jest W2V i MERT (InstanceNorm1d na każdej modalności, wymiar `[B, T, 1024]`). Dopiero potem następuje concat i dalszy forward.

**Różnica vs v2:** v2 normalizuje już połączony tensor 2048-d; v5 normalizuje każdą modalność z osobna, zachowując ich rozdzielenie do momentu fuzji.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.172 | 0.364 |
| F1 | 0.831 | 0.711 |
| Precision | 0.763 | 0.598 |
| Recall | 0.913 | 0.875 |

---

### v6 — early GMU per-feature

**Model:** `BidirectionalMambaModel`, **`fusion_type: gmu`**, `fusion_dim=1024`.

**Concat vs GMU (early fusion):**
- **Concat** skleja W2V i MERT w `[B, T, 2048]` — enkoder widzi obie modalności naraz, bez uczenia wagi między nimi przed Mambą.
- **GMU (Gated Multimodal Unit)** — moduł `GMU_per_feature` (`fusion.py`):
  1. W2V i MERT projektowane osobno: `h_w2v = tanh(W_w2v · x_w2v)`, `h_mert = tanh(W_mert · x_mert)`.
  2. Bramka: `z = σ(W_gate · [x_w2v; x_mert])` — **1024 niezależnych wartości** sigmoid na każdy krok czasu (po jednej na wymiar fuzji).
  3. Wyjście: `z ⊙ h_w2v + (1 − z) ⊙ h_mert` → `[B, T, 1024]`.

Każdy wymiar reprezentacji fuzji ma własną bramkę decydującą, ile bierze z W2V vs MERT. Enkoder BiMamba dostaje `[B, T, 1024]` zamiast 2048-d concat.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.457 | 0.324 |
| F1 | 0.671 | 0.740 |
| Precision | 0.577 | 0.617 |
| Recall | 0.803 | 0.925 |

---

### v7 — early Scalar GMU

**Model:** `BidirectionalMambaModel`, **`fusion_type: scalar-gmu`**, `fusion_dim=1024`.

**Różnica vs v6 (GMU per-feature):**
- **Scalar GMU** (`ScalarGMU`) — ta sama formuła `z · h_w2v + (1 − z) · h_mert`, ale bramka `z` ma wymiar **1 na krok czasu** (jedna wartość sigmoid na cały wektor 1024-d w danym `t`).
- Wszystkie wymiary fuzji w danym momencie dzielą tę samą proporcję W2V/MERT.
- Mniej parametrów w warstwie bramkującej niż GMU per-feature.

Fuzja nadal **przed** enkoderem BiMamba (early fusion).

| | MoM | M6 |
|---|-----|-----|
| EER | 0.169 | 0.437 |
| F1 | 0.823 | 0.676 |
| Precision | 0.718 | 0.537 |
| Recall | 0.963 | 0.913 |

---

### late-fusion v1 — osobne enkodery + late concat

**Model:** `LateFusionBidirectionalMambaModel`, **`fusion_type: concat` (late)**, `norm_type: none`.

**Early vs late fusion:**
- **Early fusion (v1–v7, v9):** W2V i MERT łączone (concat lub GMU) **przed** jednym wspólnym BiMamba.
- **Late fusion:** każda modalność ma **własny** stos Mamba:
  - W2V → `linear_forward_w2v` → Mamba forward → `[B, T, d_model]`
  - W2V (flip) → `linear_backward_w2v` → Mamba backward → flip → `[B, T, d_model]`
  - analogicznie dla MERT (osobne 4 stosy: W2V fwd/bwd, MERT fwd/bwd)
- Reprezentacje po enkodowaniu: `out_w2v = [fwd_w2v ‖ bwd_w2v]` → `[B, T, 2·d_model]`, to samo dla MERT.
- **Late concat:** `out = [out_w2v ‖ out_mert]` → `[B, T, 4·d_model]` → mean po czasie → `Linear(4·d_model → 1)`.

Modalności przechodzą przez niezależne ścieżki temporalne; concat następuje na wyższym poziomie abstrakcji.

**Hiperparametry:** `d_model=512`, `n_layers=2`, `dropout=0.3`, `label_smoothing=0.1`.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.142 | 0.238 |
| F1 | 0.821 | 0.726 |
| Precision | 0.726 | 0.583 |
| Recall | 0.946 | 0.962 |

---

### v8 — bez label smoothing

**Model:** early fusion, concat — konfiguracja zbliżona do v4 (`norm_type: none`, pełne augmentacje).

**Zmiana:** **`label_smoothing: 0.0`** — zamiast `SmoothedBCEWithLogitsLoss` (smoothing 0.1) używany jest zwykły `BCEWithLogitsLoss`.

**Label smoothing:** etykiety twarde (0/1) zastępowane są miękkimi (np. fake → 0.05, real → 0.95), co ogranicza pewność predykcji i podnosi dolne ograniczenie loss (~0.1985 przy smoothing 0.1 i idealnym dopasowaniu).

| | MoM | M6 |
|---|-----|-----|
| EER | 0.192 | 0.426 |
| F1 | 0.752 | 0.702 |
| Precision | 0.609 | 0.545 |
| Recall | 0.983 | 0.988 |

---

### v9 — mniejszy model

**Model:** `BidirectionalMambaModel`, early fusion, concat, `norm_type: none`.

**Zmiana względem v4:** redukcja pojemności enkodera:
- `d_model`: 512 → **256**
- `n_layers`: 4 → **2**
- `dropout`: 0.3 → **0.5**
- `weight_decay`: 0.01 → **0.05**

Mniejszy wspólny BiMamba po early concat — test, czy mniejsza architektura działa jak regularizacja.

| | MoM | M6 |
|---|-----|-----|
| EER | 0.261 | 0.511 |
| F1 | 0.720 | 0.693 |
| Precision | 0.571 | 0.534 |
| Recall | 0.973 | 0.988 |

---

## 8. Surowe wyniki JSON (archiwum)

<details>
<summary>Kliknij, aby rozwinąć</summary>

**v1 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.194,0.730,0.586,0.970]]}`  
**v1 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.265,0.699,0.537,1.000]]}`

**v2 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.344,0.680,0.589,0.806]]}`  
**v2 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.364,0.688,0.604,0.800]]}`

**v3 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.534,0.432,0.521,0.370]]}`  
**v3 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.590,0.187,0.370,0.125]]}`

**v4 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.158,0.836,0.765,0.921]]}`  
**v4 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.344,0.747,0.627,0.925]]}`

**v5 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.172,0.831,0.763,0.913]]}`  
**v5 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.364,0.711,0.598,0.875]]}`

**v6 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.457,0.671,0.577,0.803]]}`  
**v6 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.324,0.740,0.617,0.925]]}`

**v7 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.169,0.823,0.718,0.963]]}`  
**v7 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.437,0.676,0.537,0.913]]}`

**late-fusion MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.142,0.821,0.726,0.946]]}`  
**late-fusion M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.238,0.726,0.583,0.962]]}`

**v8 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.192,0.752,0.609,0.983]]}`  
**v8 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.426,0.702,0.545,0.988]]}`

**v9 MoM:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.261,0.720,0.571,0.973]]}`  
**v9 M6:** `{"columns":["dataloader","eer","f1","precision","recall"],"data":[[0,0.511,0.693,0.534,0.988]]}`

</details>
