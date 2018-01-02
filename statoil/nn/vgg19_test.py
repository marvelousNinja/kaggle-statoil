from datetime import datetime
from keras.callbacks import TensorBoard
from keras import optimizers
import numpy as np
from statoil.nn.vgg19 import build_model
from statoil.shared.data import load_train
from statoil.shared.data import convert_to_normalized_array

def test_vgg19():
    data = load_train()
    r = convert_to_normalized_array(data['band_1']).reshape(-1, 75, 75)
    g = convert_to_normalized_array(data['band_2']).reshape(-1, 75, 75)
    b = r / (g + 0.01)
    X = np.dstack([r, g, b]).reshape(-1, 75, 75, 3)
    y = data['is_iceberg']

    X_tr, y_tr = X[:1408], y[:1408]
    X_val, y_val = X[1408:], y[1408:]

    model = build_model()

    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.Adadelta(),
        metrics=['binary_accuracy'])

    model.fit(
        X_tr, y_tr,
        batch_size=128,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[TensorBoard(log_dir='/tmp/vgg19/run-{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))])
