from datetime import datetime
from keras.callbacks import TensorBoard
from keras import optimizers
import numpy as np
from skimage.transform import resize
from statoil.nn.resnet50 import build_model
from statoil.shared.data import load_train
from statoil.shared.data import convert_to_normalized_array

def test_resnet50():
    data = load_train()
    r = convert_to_normalized_array(data['band_1']).reshape(-1, 75, 75)
    g = convert_to_normalized_array(data['band_2']).reshape(-1, 75, 75)
    b = r / (g + 0.01)
    X = np.dstack([r, g, b]).reshape(-1, 75, 75, 3)
    y = data['is_iceberg']

    X_resized = []
    for i in range(len(X)):
        new_img = resize(X[i], (197, 197))
        X_resized.append(new_img)

    X_resized = np.concatenate(X_resized).reshape(-1, 197, 197, 3)

    X_tr, y_tr = X_resized[:1408], y[:1408]
    X_val, y_val = X_resized[1408:], y[1408:]

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
        callbacks=[TensorBoard(log_dir='/tmp/resnet50/run-{}'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))])
