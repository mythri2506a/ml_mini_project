import pandas as pd

tracks = pd.read_csv("data/fma_metadata/tracks.csv", index_col=0, header=[0, 1])
features = pd.read_csv("data/fma_metadata/features.csv", index_col=0, header=[0, 1, 2])

small = tracks[tracks[("set", "subset")] == "small"]
listens = small[("track", "listens")]
y = (listens >= listens.quantile(0.75)).astype(int)   # top 25% = "hit"

X = features.loc[small.index]
X.columns = ["_".join(c) for c in X.columns]
print(X.shape, y.value_counts())

out = X.copy()
out["artist_id"] = small[("artist", "id")]   # needed for artist-wise splitting
out["hit"] = y
out.to_csv("data/dataset.csv")
print("Saved data/dataset.csv")