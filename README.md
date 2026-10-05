# Predicting Hit Songs from Audio Features

**UE24CS352A Machine Learning — Mini-Project**

| Team member | SRN |
|---|---|
| Mythri | PES1UG24AM419 |
| Manya | PES1UG24AM365 |

## Problem statement

Songwriters believe a catchy hook (usually the chorus) is what makes a song popular. Based on the CS229 project *"Predicting Hit Songs Using Repeated Chorus"* (Liu, 2021), we test this idea: **can a model predict whether a song is a hit using only audio features from a short excerpt of the song?**

## Dataset

We use the **Free Music Archive (FMA)** dataset (Defferrard et al., 2017), `fma_small` subset.

- **8,000 tracks** from 2,309 artists, each a 30-second mid-song excerpt
- **518 audio features** per track: 11 librosa features (MFCC, chroma STFT/CQT/CENS, tonnetz, spectral centroid/bandwidth/contrast/rolloff, RMS energy, zero-crossing rate), each summarised with 7 statistics (mean, std, skew, kurtosis, median, min, max). This is the same feature set used in the original paper.
- **Label:** a track is a **hit (1)** if its FMA listen count is in the top 25%, otherwise **not a hit (0)**. This gives 2,000 hits and 6,000 non-hits.

**Differences from the original paper:** the paper used 554 Billboard songs with an extracted chorus. We use FMA because it is 14× larger and freely licensed. Our excerpt is a mid-song clip rather than a detected chorus, and "hit" means popular on FMA rather than charting on Billboard.

## Approach

1. **Artist-wise train/test split (80/20).** No artist appears in both sets, so the model is tested on artists it has never seen and cannot simply learn "popular artist = hit".
2. **Pipeline:** standardisation → PCA (95% variance) → classifier. Scaling and PCA are fitted inside each cross-validation fold to avoid data leakage. Tree models use the raw 518 features.
3. **Models:** Logistic Regression (elastic net), LDA (shrinkage), Linear / RBF / Polynomial SVM, Random Forest, Gradient Boosting, Neural Network (MLP with L2 weight decay).
4. **Tuning:** grid search with 5-fold **GroupKFold** (grouped by artist), optimising ROC-AUC. Class imbalance is handled with balanced class weights.
5. **Evaluation:** ROC-AUC and F1 on the held-out test set, compared with two baselines (always predict "not hit", and random guessing). Accuracy alone is misleading here because 75% of songs are non-hits.

## Results (held-out test set, unseen artists)

| Model | CV ROC-AUC | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|---|
| Baseline: always "not hit" | – | 0.776 | 0.000 | 0.500 |
| Baseline: random guess | – | 0.628 | 0.226 | 0.491 |
| LR (elastic net) | 0.697 | 0.611 | 0.414 | 0.661 |
| LDA (shrinkage) | 0.697 | 0.766 | 0.281 | 0.671 |
| Linear SVM | 0.697 | 0.607 | 0.413 | 0.660 |
| **RBF SVM** (selected by CV) | **0.711** | 0.666 | 0.413 | 0.664 |
| Poly SVM | 0.652 | 0.702 | 0.394 | 0.643 |
| Random Forest | 0.704 | 0.792 | 0.255 | 0.691 |
| Gradient Boosting | 0.702 | 0.704 | **0.443** | **0.695** |
| Neural Network | 0.674 | 0.739 | 0.262 | 0.647 |

**Key findings**

- Every model beats both baselines (ROC-AUC 0.64–0.70 vs 0.50), so audio does carry real information about popularity, but the signal is weak.
- **Accuracy is misleading:** Random Forest has the highest accuracy (0.792), yet it finds only 16% of hits (recall 0.159), barely better than always saying "not hit" (0.776).
- CV and test scores are close, so the models are not overfitting.
- **PCA:** 205 of 518 components explain 95% of the variance.
- **Feature importance** (drop in test ROC-AUC when a feature family is shuffled): MFCC matters most (0.085), followed by chroma STFT (0.039) and spectral contrast (0.030). The rest have almost no effect. These are mainly timbre features, which describe the overall sound and production of a song.

**Conclusion:** the "hook" idea is partly supported. Audio from a short excerpt predicts popularity better than chance, but most of a song's success depends on factors outside the audio. A limitation is that timbre features also capture genre, so part of the signal may come from some genres being more popular on FMA.

## Repository structure

```
├── src/
│   ├── prepare_data.py   # load FMA metadata + features, create hit labels → data/dataset.csv
│   ├── train.py          # train and tune all models → results/results.csv, best_model.joblib
│   ├── make_plots.py     # figures for the report → results/figures/
│   └── predict.py        # demo: predict hit / not hit for any audio file
├── results/
│   ├── results.csv
│   ├── feature_importance.csv
│   ├── best_model.joblib
│   └── figures/
├── requirements.txt
└── README.md
```

The `data/` folder is not in the repository because the files are too large. Download them as described below.

## Setup

Requires Python 3.11 or newer.

```
git clone <this-repo-url>
cd ml_mini_project
python -m pip install -r requirements.txt
```

**Download the data:** get `fma_metadata.zip` (342 MB) from https://github.com/mdeff/fma, then extract it so that these files exist:

```
data/fma_metadata/tracks.csv
data/fma_metadata/features.csv
```

## How to run

Run all commands from the project root folder.

```
python src/prepare_data.py   # ~30 s   → data/dataset.csv
python src/train.py          # ~15 min → results/results.csv, results/best_model.joblib
python src/make_plots.py     # ~15 min → results/figures/*.png
```

**Demo: predict for any song**

```
python src/predict.py "path/to/song.mp3"
python src/predict.py "path/to/song.mp3" --start 60   # use the 30 s starting at 1:00
```

The script takes a 30-second excerpt (from the middle of the song by default), computes the same 518 features as FMA, and prints **HIT** or **NOT A HIT** with a decision score (above 0 leans towards hit).

To try it without your own music, librosa can download a free sample song:

```
python -c "import librosa; print(librosa.ex('vibeace'))"
```

## References

1. Liu, E. (2021). *Predicting Hit Songs Using Repeated Chorus.* CS229 project report, Stanford University.
2. Defferrard, M., Benzi, K., Vandergheynst, P., & Bresson, X. (2017). *FMA: A Dataset for Music Analysis.* ISMIR.
3. McFee, B. et al. *librosa: Audio and music signal analysis in Python.* https://librosa.org
4. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR.
