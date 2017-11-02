import numpy as np
import pandas as pd
import random as rd

from sklearn.model_selection import train_test_split

class DataGenerator():
    def __init__(self, test=False, batch_size=64):
        self.batch_size = batch_size
        self.train_json_path = 'data/train.json'
        self.test_json_path = 'data/test.json'

        samples = pd.read_json(self.train_json_path)
        band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in samples["band_1"]])
        band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in samples["band_2"]])
        samples_x = np.concatenate([band1[:, :, :, np.newaxis], band2[:, :, :, np.newaxis]], axis=-1)
        samples_y = np.array(samples["is_iceberg"])

        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
            samples_x, samples_y, random_state=151018, train_size=0.75)

        if test:
            test_samples = pd.read_json(self.test_json_path)
            test_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_samples["band_1"]])
            test_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_samples["band_2"]])
            self.test_x = np.concatenate([test_band1[:, :, :, np.newaxis], test_band2[:, :, :, np.newaxis]], axis=-1)

    def train_generator(self):
        pass

    def validate_generator(self):
        pass

    def get_validate_batch_count(self):
        pass

    def test_generator(self):
        pass
