from __future__ import absolute_import

import os
import numpy as np
import sys

import keras
import tensorflow as tf
from keras.initializers import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
import keras.backend as K

from data_provider import (
    BD2013_training_data,
    DataGenerator,
    BD2013_HLA_filter_by_length_list,
    split_samples,
)
from model import (
    model_config,
)


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
BATCH_SIZE = 16


def train_2013():
    model = model_config()
    model.compile(optimizer=SGD(lr=0.001, momentum=0.8),
                  loss=["mean_squared_error", "binary_crossentropy"])

    # setup data for generators
    samples = np.array(BD2013_HLA_filter_by_length_list([9]))
    train_samples, validate_samples = split_samples(
        samples, validate_ratio=0.2)
    train_generator = DataGenerator(BATCH_SIZE, train_samples)
    validate_generator = DataGenerator(
        BATCH_SIZE, validate_samples, validate=True)
    model.train_generator = train_generator
    model.validate_generator = validate_generator

    best_model_save_path = os.path.join(BASE_DIR, 'best_model_2013.keras')

    model.fit_generator(model.train_generator,
                        epochs=1000,
                        validation_data=model.validate_generator,
                        steps_per_epoch=len(model.train_generator),
                        validation_steps=len(model.validate_generator),
                        callbacks=[
                            ReduceLROnPlateau(
                                monitor='val_loss',
                                factor=0.5,
                                patience=5,
                                cooldown=3,
                                min_lr=0.00001),
                            TensorBoard(log_dir=os.path.join(BASE_DIR,
                                                             'logs_MULTI')),
                            ModelCheckpoint(best_model_save_path,
                                            monitor='val_loss',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                            mode='auto',
                                            period=1),
                            EarlyStopping(monitor='val_loss',
                                          patience=15,
                                          verbose=1,
                                          min_delta=0.001),
                        ])


if __name__ == '__main__':
    pass
