import numpy as np
import pandas as pd
from PIL import Image

train = pd.read_json('./data/train.json')


# TODO AS: Global scaling
scale = lambda a: ((a - a.min()) / (a.max() - a.min()) * 255).astype('uint8')
min_db = -50
max_db = 40
signal_scale = lambda a: ((a - min_db) / (max_db - min_db) * 255).astype('uint8')

i = 0
for _, record in train.iterrows():
    print('Saving ', i)
    i += 1
    record_id = record['id']
    label = 'iceberg' if record['is_iceberg'] else 'ship'

    hh = np.array(record['band_1']).reshape(75, 75)
    hv = np.array(record['band_2']).reshape(75, 75)
    div = hh / hv

    composites = {
        'hh': signal_scale(hh).reshape(75, 75),
        'hv': signal_scale(hv).reshape(75, 75),
        'div': scale(div).reshape(75, 75),
        'hh_hh_hv': np.dstack([signal_scale(hh), signal_scale(hh), signal_scale(hv)]),
        'hh_hv_hv': np.dstack([signal_scale(hh), signal_scale(hv), signal_scale(hv)]),
        'hh_hv_div': np.dstack([signal_scale(hh), signal_scale(hv), scale(div)])
    }

    for version, data in composites.items():
        if len(data.shape) > 2:
            img = Image.fromarray(data, 'RGB')
        else:
            img = Image.fromarray(data).convert('RGB')
        img.save('./data/images/train/{}/{}_{}.png'.format(label, record_id, version))
