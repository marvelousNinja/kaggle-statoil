import numpy as np
import pandas as pd
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Cropping2D
from keras.models import  Sequential
from keras.callbacks import TensorBoard

autoencoder = Sequential([
    # Encoder
    Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(75, 75, 1)),
    MaxPool2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    MaxPool2D((2, 2), padding='same'),
    # Decoder
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(8, (3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', padding='valid'),
    UpSampling2D((2, 2)),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
    Cropping2D(((1, 0), (1, 0)))
])

print(autoencoder.summary())

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

def prep_image(data):
    data = np.array(data)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data.reshape(75, 75).reshape(-1)

train = pd.read_json('./data/train.json')
test = pd.read_json('./data/test.json')
all_images = train.band_2.append(test.band_2)
prepped_images = np.concatenate(all_images.apply(prep_image).values).reshape(-1, 75, 75, 1).astype('float32')

X_train = prepped_images[:7936]
X_test = prepped_images[7936:]

autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(X_test, X_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
