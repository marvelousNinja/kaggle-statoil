from keras.applications import InceptionV3
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from statoil.util.transfer_learning import drop_n_and_freeze

def build_model():
    return Sequential([
        drop_n_and_freeze(0, InceptionV3(include_top=False, input_shape=(139, 139, 3))),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')])
