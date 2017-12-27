import numpy as np
import pandas as pd
from PIL import Image

train = pd.read_json('./data/train.json')

i = 0
for _, record in train.iterrows():
    print('Saving ', i)
    i += 1
    record_id = record['id']

    band_1_data = np.array(record['band_1'])
    band_2_data = np.array(record['band_2'])
    band_1_data = (band_1_data - np.min(band_1_data)) / (np.max(band_1_data) - np.min(band_1_data)) * 255
    band_2_data = (band_2_data - np.min(band_2_data)) / (np.max(band_1_data) - np.min(band_1_data)) * 255
    band_1_data = band_1_data.reshape(75, 75)[25:50, 25:50]
    band_2_data = band_2_data.reshape(75, 75)[25:50, 25:50]

    band_1 = Image.fromarray(band_1_data).convert('RGB')
    band_2 = Image.fromarray(band_2_data).convert('RGB')
    label = 'iceberg' if record['is_iceberg'] else 'ship'
    band_1.save('./data/images/train/{}/{}_band_1.png'.format(label, record_id))
    band_2.save('./data/images/train/{}/{}_band_2.png'.format(label, record_id))
