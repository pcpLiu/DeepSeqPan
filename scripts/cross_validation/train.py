"""

Use mean_squared_error

VGG
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = ""   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import math

import numpy as np




from data_provider import (
    BD2013_training_data,
    DataGenerator,
    BD2013_HLA_filter_by_length_list_CDHIT,
    BD2013_HLA_filter_by_length_list,
    split_samples,
    DATA_ENTRY,
    valid_allele_list,
    hla_encode_ONE_HOT,
    encode_ligand,
    )

from model import (
    model_config,
)

import keras
import sys
import tensorflow as tf
from keras.initializers import *
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.utils import multi_gpu_model
import keras.backend as K

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class TrainVlidateReSplit(keras.callbacks.Callback):
    """
    A callback to re-split train and validate data every 5 epochs
    """
    def __init__(self, samples, patience=10, min_delta=0.0):
        super(TrainVlidateReSplit, self).__init__()
        self.samples = samples
        self.patience = patience
        self.last_update_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.last_update_epoch + self.patience:
                return

        print(" [TrainVlidateReSplit]: Epoch {}, re-split train/validate samples.".format(epoch + 1, self.patience))
        train_samples, validate_samples = split_samples(self.samples, validate_ratio=0.2)
        self.model.train_generator.samples = train_samples
        self.model.train_generator.init_data()

        self.model.validate_generator.samples = validate_samples
        self.model.validate_generator.init_data()

        self.last_update_epoch = epoch # reset patience count


def train_fold(samples, test_samples, cv_fold, out_file):
    print("Train on", len(samples), ", test on", len(test_samples))
    model = model_config()
    model.compile(optimizer=SGD(lr=0.001, momentum=0.8),
        loss=["mean_squared_error", "binary_crossentropy"])

    # setup data for generators
    train_samples, validate_samples = split_samples(samples, validate_ratio=0.2)
    train_generator = DataGenerator(256, train_samples)
    validate_generator = DataGenerator(256, validate_samples, validate=True)
    model.train_generator = train_generator
    model.validate_generator = validate_generator

    best_model_save_path = os.path.join(BASE_DIR, 'best_model_cv_{}.keras'.format(cv_fold))

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

    # load best
    model = load_model(best_model_save_path)

    # predict
    allele_list = valid_allele_list()
    proteins_features = []
    ligand_features = []
    logic50s = []
    for sample in test_samples:
        sample = DATA_ENTRY(*sample)
        mhc = sample.mhc
        if mhc not in allele_list:
            continue

        proteins_features.append(hla_encode_ONE_HOT(mhc))
        ligand_features.append(encode_ligand(sample.sequence))
        logic50s.append(math.log(float(sample.ic50)))

    results = model.predict({
                    'protein': np.array(proteins_features),
                    'ligand': np.array(ligand_features)
                    })

    results = np.array(results)
    print(results.shape)
    # write results
    for i in range(len(logic50s)):
        real = logic50s[i]
        predict_ic = results[0][i][0]
        predict_binding = results[1][i][0]
        print("real: {}, predict_ic: {}, preidct_binding: {}".format(real, predict_ic, predict_binding))
        sample = DATA_ENTRY(*(test_samples[i]))
        out_file.write("{},{},{},{},{}\n".format(sample.mhc, sample.sequence, real, predict_ic, predict_binding))

    print("Finish fold {}...".format(i))
    print('\n'*8)


def train_cv():
    from sklearn.model_selection import KFold

    # get samples
    if sys.argv[1].upper() == 'CDHIT':
        print('CV on cdhit filtered BD2013')
        samples = BD2013_HLA_filter_by_length_list_CDHIT([9])
    else:
        print('CV on raw BD2013')
        samples = BD2013_HLA_filter_by_length_list([9])

    allele_list = valid_allele_list()
    samples = np.array(list(filter(lambda x: x.mhc in allele_list, samples)))
    for _ in range(4):
        np.random.shuffle(samples)

    FOLD = int(sys.argv[2])

    kf = KFold(n_splits=FOLD, shuffle=True)
    i = 0
    out_file = open('cv_result.txt', 'w')
    out_file.write("hla,sequence,real_log,pred_log,pred_binding\n")
    for train, test in kf.split(samples):
        print("Fold: ", i)
        train_fold(samples[train], samples[test], i, out_file)
        i += 1

if __name__ == '__main__':
    train_cv()
