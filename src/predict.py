"""
Live demo: predict whether a song is a "hit" from a short audio excerpt.

Usage (run from the project root):
    python src/predict.py path/to/song.mp3
    python src/predict.py path/to/song.mp3 --start 45      # use the 30s starting at 0:45
    python src/predict.py data/fma_small/000/000002.mp3 --compare 2

The features are computed exactly the way the FMA dataset computed them
(same 11 librosa features x 7 statistics = 518 values), so a new song is
described in the same "language" the model was trained on.

--compare TRACK_ID checks our extracted features against FMA's precomputed
features.csv for that track (a sanity check worth showing in the review).
"""
import argparse
import warnings

import joblib
import librosa
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

CLIP_SECONDS = 30  # FMA's features were computed on 30-second clips


def load_clip(path, start=None):
    """Load a 30s mono clip. Default: centred in the song (FMA clips are mid-song)."""
    total = librosa.get_duration(path=path)
    if start is None:
        start = max(0.0, total / 2 - CLIP_SECONDS / 2)
    y, sr = librosa.load(path, sr=None, mono=True, offset=start, duration=CLIP_SECONDS)
    return y, sr, start, total


def extract_features(y, sr):
    """Re-implementation of FMA's features.py, updated for modern librosa."""
    feats = {}

    def add_stats(name, values):
        summaries = {
            "mean": np.mean(values, axis=1),
            "std": np.std(values, axis=1),
            "skew": stats.skew(values, axis=1),
            "kurtosis": stats.kurtosis(values, axis=1),
            "median": np.median(values, axis=1),
            "min": np.min(values, axis=1),
            "max": np.max(values, axis=1),
        }
        for stat, arr in summaries.items():
            arr = np.nan_to_num(arr)  # skew/kurtosis are NaN if a feature is constant (e.g. silence)
            for i, v in enumerate(arr):
                feats[f"{name}_{stat}_{i + 1:02d}"] = float(v)

    add_stats("zcr", librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512))

    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12))
    add_stats("chroma_cqt", librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7))
    cens = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    add_stats("chroma_cens", cens)
    add_stats("tonnetz", librosa.feature.tonnetz(chroma=cens))

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    add_stats("chroma_stft", librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12))
    add_stats("rmse", librosa.feature.rms(S=stft))  # FMA calls it "rmse"
    add_stats("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
    add_stats("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
    add_stats("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
    add_stats("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    add_stats("mfcc", librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20))

    return pd.Series(feats)


def align_to_model(feats, model):
    """Put features in the exact column order the model was trained with."""
    expected = list(model.feature_names_in_)
    missing = [c for c in expected if c not in feats.index]
    if missing:
        raise ValueError(f"{len(missing)} features missing, e.g. {missing[:3]}")
    return pd.DataFrame([feats[expected].values], columns=expected)


def compare_with_fma(feats, track_id, path="data/fma_metadata/features.csv"):
    """Compare our features with FMA's precomputed ones for the same track."""
    print(f"\nComparing with FMA's features.csv for track {track_id} (takes ~20s) ...")
    ref = pd.read_csv(path, index_col=0, header=[0, 1, 2])
    ref.columns = ["_".join(c) for c in ref.columns]
    if track_id not in ref.index:
        print("   Track not found in features.csv.")
        return
    ref_row = ref.loc[track_id]
    ours = feats[ref_row.index]
    corr = np.corrcoef(ours.values, ref_row.values)[0, 1]
    print(f"   Correlation between our 518 features and FMA's: {corr:.4f}")
    print("   (Close to 1.0 means our extraction matches the training data.)")


def main():
    parser = argparse.ArgumentParser(description="Predict if a song is a hit from its audio.")
    parser.add_argument("audio", help="path to an audio file (mp3, wav, ...)")
    parser.add_argument("--start", type=float, default=None,
                        help="start of the 30s excerpt in seconds (default: middle of song)")
    parser.add_argument("--model", default="results/best_model.joblib")
    parser.add_argument("--compare", type=int, default=None, metavar="TRACK_ID",
                        help="FMA track id to compare extracted features against")
    args = parser.parse_args()

    model = joblib.load(args.model)
    final_step = model.steps[-1][1] if hasattr(model, "steps") else model
    print(f"Model: {type(final_step).__name__}")

    y, sr, start, total = load_clip(args.audio, args.start)
    print(f"Song length {total:.0f}s | analysing {start:.0f}s to {start + len(y) / sr:.0f}s")

    feats = extract_features(y, sr)
    X = align_to_model(feats, model)
    is_hit = int(model.predict(X)[0])

    print("\n" + "=" * 40)
    print(f"  Prediction: {'HIT' if is_hit else 'NOT A HIT'}")
    if hasattr(model, "predict_proba"):
        print(f"  Hit probability: {model.predict_proba(X)[0, 1]:.1%}")
    else:
        score = model.decision_function(X)[0]
        print(f"  Decision score: {score:+.3f}  (above 0 = hit side, further = more confident)")
    print("=" * 40)

    if args.compare is not None:
        compare_with_fma(feats, args.compare)


if __name__ == "__main__":
    main()
