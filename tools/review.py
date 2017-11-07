import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

train = pd.read_json('data/train.json')

while True:
    index = rd.randint(0, len(train)-1)
    band_1 = np.reshape(train['band_1'][index], (75, 75))
    band_2 = np.reshape(train['band_2'][index], (75, 75))

    print band_1

    print index, train['is_iceberg'][index], train['inc_angle'][index]
    plt.subplot(121)
    plt.imshow(band_1)
    plt.subplot(122)
    plt.imshow(band_2)
    plt.show()


