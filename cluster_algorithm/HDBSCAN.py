
from sklearn.datasets import make_blobs
import pandas as pd
import hdbscan


blobs, labels = make_blobs(n_samples=2000, n_features=10)
clusterer = hdbscan.HDBSCAN()
clusterer.fit(blobs)






