from keras.applications import ResNet50
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from statoil.util.transfer_learning import drop_n_and_freeze

def build_model():
    return Sequential([
        drop_n_and_freeze(0, ResNet50(include_top=False, input_shape=(197, 197, 3))),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')])
