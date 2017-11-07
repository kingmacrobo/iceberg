import numpy as np
import pandas as pd
import random as rd

from sklearn.model_selection import train_test_split

class DataGenerator():
    def __init__(self, test=False, batch_size=64):
        self.batch_size = batch_size
        self.train_json_path = 'data/train.json'
        self.test_json_path = 'data/test.json'

        print 'Load data from json file...'
        samples = pd.read_json(self.train_json_path)
        band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in samples["band_1"]])
        band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in samples["band_2"]])
        samples_x = np.concatenate([band1[:, :, :, np.newaxis], band2[:, :, :, np.newaxis]], axis=-1)
        samples_y = np.array(samples["is_iceberg"])

        print 'Split train data to train and val...'
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(
            samples_x, samples_y, random_state=151018, train_size=0.75)

        if test:
            test_samples = pd.read_json(self.test_json_path)
            test_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_samples["band_1"]])
            test_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_samples["band_2"]])
            self.test_x = np.concatenate([test_band1[:, :, :, np.newaxis], test_band2[:, :, :, np.newaxis]], axis=-1)

    def train_generator(self):
        indicies = range(len(self.train_x))
        while True:
            choice = rd.sample(indicies, self.batch_size)
            batch_x = self.train_x[choice]
            batch_y = self.train_y[choice]

            yield batch_x, batch_y


    def validate_generator(self):
        batches = self.get_validate_batch_count()
        for batch in range(batches):
            start = batch * self.batch_size
            end = min((batch + 1) * self.batch_size, len(self.val_x))
            batch_val_x = self.val_x[start: end]
            batch_val_y = self.val_y[start: end]

            yield  batch_val_x, batch_val_y

    def get_validate_batch_count(self):
        return len(self.val_x) / self.batch_size + 1 * (len(self.val_x) % self.batch_size)

    def test_generator(self):
        batches = self.get_test_batch_count()
        for batch in range(batches):
            start = batch * self.batch_size
            end = min((batch + 1) * self.batch_size, len(self.test_x))
            batch_test_x = self.test_x[start: end]

            yield  batch_test_x

    def get_test_batch_count(self):
        return len(self.test_x) / self.batch_size + 1 * (len(self.test_x) % self.batch_size)
