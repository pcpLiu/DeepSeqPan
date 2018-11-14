import os
import random
import math
import numpy as np
from collections import namedtuple


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_ENTRY = namedtuple("data_entry", "species, mhc, pep_length, sequence, ineq, ic50")

AA_SYMOLS = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']


def valid_allele_list():
    l = list()
    f = os.path.join(BASE_DIR, 'mhc_i_protein_sequence.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num > 0:
                allele = line.strip('\n').split()[0]
                l.append(allele)
    return l


def parse_data_file_BD2013():
    """
    Parse SEQUENCE_BINDING_DATA_FILE to get list of all data entry
    """
    data = list()
    f = os.path.join(BASE_DIR, 'bdata.20130222.mhci.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num > 0:
                info = line.strip('\n').split('\t')
                data.append(DATA_ENTRY(species=info[0],
                    mhc=info[1],
                    pep_length=int(info[2]),
                    sequence=info[3],
                    ineq=info[4],
                    ic50=float(info[5])))
    return data

def parse_data_file_BD2013_CDHIT():
    """
    Parse SEQUENCE_BINDING_DATA_FILE to get list of all data entry
    """
    data = list()
    f = os.path.join(BASE_DIR, 'bdata.20130222.mhci.VALID.9.CDHIT.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num > 0:
                info = line.strip('\n').split('\t')
                data.append(DATA_ENTRY(species=info[0],
                    mhc=info[1],
                    pep_length=int(info[2]),
                    sequence=info[3],
                    ineq=info[4],
                    ic50=float(info[5])))
    return data


def BD2013_HLA_filter_by_length_list(length_list):
    """BD2013 HLA filter by length
    """
    data = parse_data_file_BD2013()
    return list(filter(lambda x: 'HLA' in x.mhc and x.pep_length in length_list , data))

def BD2013_HLA_filter_by_length_list_CDHIT(length_list):
    """BD2013 HLA filter by length
    """
    data = parse_data_file_BD2013_CDHIT()
    return list(filter(lambda x: 'HLA' in x.mhc and x.pep_length in length_list , data))

def BD2013_HLA_allele_list(length_list):
    data = parse_data_file_BD2013()
    all_allele = list(set([x.mhc for x in data]))
    return list(set(list(filter(lambda x: 'HLA' in x, all_allele))))

def BD2013_training_data():
    training_ligands = BD2013_HLA_filter_by_length_list([9])
    allele_list = valid_allele_list()

    hla_values = []
    ligand_values = []
    log_ic50_values = []
    for entry in training_ligands:
        mhc = entry.mhc
        if mhc not in allele_list:
            continue

        # protein
        hla_values.append(hla_encode_ONE_HOT(mhc))
        # ligand
        ligand_values.append(encode_ligand(entry.sequence))
        # log_ic50
        log_ic50_values.append(math.log(entry.ic50))

    return np.array(hla_values), np.array(ligand_values), np.array(log_ic50_values)


def BD2013_training_data_CDHIT():
    training_ligands = BD2013_HLA_filter_by_length_list_CDHIT([9])
    allele_list = valid_allele_list()

    hla_values = []
    ligand_values = []
    log_ic50_values = []
    for entry in training_ligands:
        mhc = entry.mhc
        if mhc not in allele_list:
            continue

        # protein
        hla_values.append(hla_encode_ONE_HOT(mhc))
        # ligand
        ligand_values.append(encode_ligand(entry.sequence))
        # log_ic50
        log_ic50_values.append(math.log(entry.ic50))

    return np.array(hla_values), np.array(ligand_values), np.array(log_ic50_values)


def encode_ligand(ligand):
    m = list()
    for symbol in AA_SYMOLS:
        channel = list()
        for aa in ligand:
            if aa.upper() == symbol: channel.append(1.0)
            else: channel.append(random.uniform(0.001, 0.01))
        m.append(channel)
    m = np.array(m).reshape(1, len(ligand), 20)
    return m

def get_hla_aligned_sequence(hla_alle):
    """
    Reading data from [mhc_i_protein_sequence.txt]
    """
    max_length = 0
    f = os.path.join(BASE_DIR, 'mhc_i_protein_sequence.txt')
    with open(f, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                max_length = int(line.strip('\n').split()[1])
            else:
                info = line.strip('\n').split()
                if info[0] == hla_alle:
                    sequence = info[1]
                    break
    return max_length, sequence


def hla_encode_ONE_HOT(hla_allele):
    """
    Give MHC allele, return one_hot encode [1, MAX_LENGTH, 21]
    """
    """
    Encode HLA protein sequence with one-hot-encoding. The dimension should be [1, MAX_LENGTH, 21].
    Any missing AA with be encoded as 21 channel

    """
    max_length, sequence = get_hla_aligned_sequence(hla_allele)
    matrix = np.zeros((1, max_length, 21))
    for i in range(max_length):
        if i < len(sequence) and sequence[i] in AA_SYMOLS:
            aa_index = AA_SYMOLS.index(sequence[i])
            matrix[0][i][aa_index] = 1.0
        else:
            matrix[0][i][-1] = 1.0
    return matrix


##############################################################################
#
# Generator with shuffle
#
#

def split_samples(samples, validate_ratio=0.2):
    for _ in range(10):
        np.random.shuffle(samples)
    train_count = math.ceil(len(samples) * (1 - validate_ratio))
    return samples[:train_count], samples[train_count:]

class DataGenerator():
    def __init__(self, batch_size, samples, validate=False):
        self.last_epoch = -1
        self.batch_size = batch_size
        self.samples = samples # may be changed
        self.allele_list = valid_allele_list()
        self.validate = validate
        self.proteins_features_map = {}
        self.ligand_feature_map = {}

        self.init_data()

    def init_data(self):
        for sample in self.samples:
            sample = DATA_ENTRY(*sample)
            mhc = sample.mhc
            if mhc not in self.allele_list:
                continue

            if mhc not in self.proteins_features_map:
                self.proteins_features_map[mhc] = hla_encode_ONE_HOT(mhc)

            if sample.sequence not in self.ligand_feature_map:
                self.ligand_feature_map[sample.sequence] = encode_ligand(sample.sequence)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)


    def __next__(self):
        protein_feature_list = []
        ligand_feature_list = []
        log_ic50_values = []
        binary_labels = []
        sampled = set()

        i = 0
        while i < self.batch_size:
            sample = self.samples[np.random.choice(np.arange(0,len(self.samples)), size=1)[0]]
            sample = DATA_ENTRY(*sample)

            if sample in sampled:
                continue

            # skip invalid mhc
            mhc = sample.mhc
            if mhc not in self.allele_list:
                continue


            # protein
            protein_feature_list.append(self.proteins_features_map[mhc])
            # ligand
            ligand_feature_list.append(self.ligand_feature_map[sample.sequence])
            # log_ic50
            log_ic50_values.append(math.log(float(sample.ic50)))
            # binary label
            binary_labels.append(0.9 if float(sample.ic50) < 500 else 0.01) #smooth labels

            sampled.add(sample)

            i += 1

        return (
                    {
                    'protein': np.array(protein_feature_list),
                    'ligand': np.array(ligand_feature_list)
                    },
                    [
                        np.array(log_ic50_values),
                        np.array(binary_labels),
                    ],
                )


if __name__ == '__main__':
    pass
