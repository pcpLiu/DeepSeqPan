import sys

import os
os.environ["CUDA_DEVICE_ORDER"] = ""   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=""

import numpy as np

from keras.models import load_model
from data_provider import (
    hla_encode_ONE_HOT,
    encode_ligand,
)

def main(model_file, testing_file):
    model = load_model(model_file)
    samples = read_file(testing_file)
    for sample in samples:
        hla = hla_encode_ONE_HOT(sample[0])
        peptide = encode_ligand(sample[1])
        out = model.predict({
            'protein': np.array([hla]),
            'ligand': np.array([peptide]),
        })
        print('{},{},{} (log_ic50),{} (binary)'.format(sample[0], sample[1], out[0][0][0], out[1][0][0]))

def read_file(testing_file):
    """
    Assume format
    ```
        allele peptide
        ....
    ```
    Only supports 9-length peptides
    """
    samples = []
    with open(testing_file, 'r') as in_file:
        for line in in_file:
            info = line.strip('\n').split()
            samples.append((info[0], info[1]))
    return samples


if __name__ == '__main__':
    model_file = sys.argv[1]
    testing_file = sys.argv[2]
    main(model_file, testing_file)
