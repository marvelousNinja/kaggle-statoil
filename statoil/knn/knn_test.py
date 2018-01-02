from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from statoil.shared.data import load_train
from statoil.shared.data import convert_to_normalized_array
from pytest import approx
import numpy as np

def test_knn():
    data = load_train()
    X = convert_to_normalized_array(data['band_1'])[:, 25:50, 25:50].reshape(-1, 25 * 25)
    y = data['is_iceberg']
    model = KNeighborsClassifier(n_neighbors=20)
    scores = cross_val_score(model, X, y, scoring='neg_log_loss')
    assert np.mean(-scores) == approx(0.48, abs=0.02)
