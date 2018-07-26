import os
import math

from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from keras.models import *

from data_provider import (
    valid_allele_list,
    all_weekly_data_filter_NEW_NONREDUNT,
    weekly_data_alleles_list_NEW_NONREDUNT,
    all_weekly_data_NEW_NONREDUNT,
    encode_ligand,
    hla_encode_ONE_HOT,
)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# result dir
TEST_RESULT_DIR = os.path.join(
    BASE_DIR, 'weekly_result_nonredundant_sep_iedbid')
if not os.path.exists(TEST_RESULT_DIR):
    os.mkdir(TEST_RESULT_DIR)

# load model
MODEL_DIR = os.path.join(BASE_DIR, 'best_model_2013.keras')
BEST_MODEL = ""


RESULT_FILE_LIST = []

DATA_TEMPLATE_NEW_FORMAT_METHODS_LIST = [
    'ConvLogIC50',
    'ConvIC50',
    'ConvBinding',
    "NetMHCpan 2.8",
    "NetMHCpan 3.0",
    "SMM",
    "NetMHC 3.4 (ANN)",
    "NetMHC 4.0 (ANN)",
    "ARB",
    "SMMPMBEC",
    "IEDB Consensus",
    "NetMHCcons",
    "PickPocket",
    "mhcflurry",
]

DATA_TEMPLATE_NEW_FORMAT = {
    'binary': {
        'count': 0,
        'real': [],
        'methods': {
            'ConvLogIC50': [],
            'ConvIC50': [],
            'ConvBinding': [],
            "NetMHCpan 2.8": [],
            "NetMHCpan 3.0": [],
            "SMM": [],
            "NetMHC 3.4 (ANN)": [],
            "NetMHC 4.0 (ANN)": [],
            "ARB": [],
            "SMMPMBEC": [],
            "IEDB Consensus": [],
            "NetMHCcons": [],
            "PickPocket": [],
            "mhcflurry": [],

        }
    },
    'ic50': {
        'count': 0,
        'real': [],
        'methods': {
            'ConvLogIC50': [],
            'ConvIC50': [],
            'ConvBinding': [],
            "NetMHCpan 2.8": [],
            "NetMHCpan 3.0": [],
            "SMM": [],
            "NetMHC 3.4 (ANN)": [],
            "NetMHC 4.0 (ANN)": [],
            "ARB": [],
            "SMMPMBEC": [],
            "IEDB Consensus": [],
            "NetMHCcons": [],
            "PickPocket": [],
            "mhcflurry": [],
        },
    },
    't1/2': {
        'count': 0,
        'real': [],
        'methods': {
            'ConvLogIC50': [],
            'ConvIC50': [],
            'ConvBinding': [],
            "NetMHCpan 2.8": [],
            "NetMHCpan 3.0": [],
            "SMM": [],
            "NetMHC 3.4 (ANN)": [],
            "NetMHC 4.0 (ANN)": [],
            "ARB": [],
            "SMMPMBEC": [],
            "IEDB Consensus": [],
            "NetMHCcons": [],
            "PickPocket": [],
            "mhcflurry": [],
        },
    },
}


def IEDB_AUC(real, predict_ic50, convert=None, reverse=True):
    """
    GET ARCC ,AUC for IEDB matrixc
    """
    real_binary = []
    if convert == 'ic50':
        for val in real:
            if val > 500.0:
                real_binary.append(0)
            else:
                real_binary.append(1)
    elif convert == 't1/2':
        for val in real:
            if val > 120.0:  # minutes
                real_binary.append(1)
            else:
                real_binary.append(0)
    else:
        # binary
        real_binary = [int(x) for x in real]

    if reverse:
        predict_ic50 = [-x for x in predict_ic50]

    fpr, tpr, _ = roc_curve(real_binary, predict_ic50)
    return auc(fpr, tpr)


def IEDB_PRCC(real, predict_ic50, convert=None, reverse=True):

    if convert == 'binary' or convert == 't1/2':
        predict_ic50 = [-x for x in predict_ic50]

    return spearmanr(real, predict_ic50)[0]


def stand_pearson(real, predict_ic50, convert=None):
    if convert == 'binary' or convert == 't1/2':
        predict_ic50 = [-x for x in predict_ic50]
    return pearsonr(real, predict_ic50)[0]


def IEDB_AUC_binary_prediction(real, predicted_binary_convincing, convert=None):
    """
    GET ARCC ,AUC for IEDB matrixc
    """
    real_binary = []
    if convert == 'ic50':
        for val in real:
            if val > 500.0:
                real_binary.append(0)
            else:
                real_binary.append(1)
    elif convert == 't1/2':
        for val in real:
            if val > 120.0:  # minutes
                real_binary.append(1)
            else:
                real_binary.append(0)
    else:
        # binary
        real_binary = [int(x) for x in real]

    fpr, tpr, _ = roc_curve(real_binary, predicted_binary_convincing)
    return auc(fpr, tpr)


def IEDB_PRCC_binary_prediction(real, predicted_binary_convincing, convert=None):
    if convert == 'ic50':
        predicted_binary_convincing = [-x for x in predicted_binary_convincing]
    return spearmanr(real, predicted_binary_convincing)[0]


def stand_pearson_binary_prediction(real, predicted_binary_convincing, convert=None):
    if convert == 'ic50':
        predicted_binary_convincing = [-x for x in predicted_binary_convincing]
    return pearsonr(real, predicted_binary_convincing)[0]


def WEEK_DATA_COUNT(real, convert):
    real_binary = []
    if convert == 'ic50':
        for val in real:
            if val > 500.0:
                real_binary.append(0)
            else:
                real_binary.append(1)
    elif convert == 't1/2':
        for val in real:
            if val > 120.0:  # minutes
                real_binary.append(1)
            else:
                real_binary.append(0)
    else:
        # binary
        real_binary = [int(x) for x in real]

    return len(real_binary), real_binary.count(1), real_binary.count(0)


def predict_on_entry(entry, hla_encoded):
    ligand = entry[5]
    ligand_encode = encode_ligand(ligand)
    result = BEST_MODEL.predict({'protein': np.array(
        [hla_encoded]), 'ligand': np.array([ligand_encode])})
    return result[0][0][0], result[1][0][0]


def test_weekly_data_on_allele(allele, test_file=None):
    # get test entryies
    entries = all_weekly_data_filter_NEW_NONREDUNT(
        [allele], [9], f_name=test_file)

    if len(entries) > 0:
        print("="*20)
        print("Test [{}] weekly data entries on allele [{}], length [9]".format(
            len(entries), allele))
        print("="*20)
    else:
        print("="*20)
        print("No weekly data entries on allele [{}], length [9]. Continue".format(
            len(entries), allele))
        print("="*20)
        return

    # get hla encoded vector
    hla_encoded = hla_encode_ONE_HOT(allele)

    # group by iedb id
    ref_ids = list(map(lambda x: x[1], entries))
    ref_ids = set(ref_ids)

    # for each date, generate separate file
    for ref_id in ref_ids:
        # out file
        title, _ = all_weekly_data_NEW_NONREDUNT()
        out_file_name = '{}_{}_weekly.txt'.format(
            allele.replace(':', '-'), ref_id)
        RESULT_FILE_LIST.append(out_file_name)
        out_file = open(os.path.join(TEST_RESULT_DIR, out_file_name), 'w')
        out_str = '{}'.format('\t'.join(title[:7]))
        out_str += '\t{}\t{}\t{}'.format('ConvLogIC50',
                                         'ConvIC50', 'ConvBinding')
        out_str += '\t{}'.format('\t'.join(title[7:]))
        out_file.write(out_str + '\n')

        # filter entries by date
        filtered_entries = list(filter(lambda x: x[1] == ref_id, entries))

        for entry in filtered_entries:
            logic50, bind_prob = predict_on_entry(entry, hla_encoded)
            out_str = '{}'.format('\t'.join(entry[:7]))
            out_str += '\t{}\t{}\t{}'.format(logic50,
                                             math.exp(logic50), bind_prob)
            out_str += '\t{}'.format('\t'.join(entry[7:]))
            out_file.write(out_str + '\n')
        out_file.close()


def get_data_from_result_file(result_file):
    """
    Get two information: binay and ic50.

    Vlues seperate by types
    """
    import copy
    print('read file', result_file)
    data = copy.deepcopy(DATA_TEMPLATE_NEW_FORMAT)

    with open(result_file, 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                title = line.strip('\n').split('\t')
                methods_name_list = title[7:]
            if line_num > 0:
                info = line.strip('\n').split('\t')
                measure_type = info[4]
                data[measure_type]['count'] += 1
                data[measure_type]['real'].append(float(info[6]))
                col_index = 0
                for val in info[7:]:
                    method_name = methods_name_list[col_index]
                    try:
                        data[measure_type]['methods'][method_name].append(
                            float(val))
                    except:
                        pass
                    col_index += 1

    return data


def pass_allle_result(result_file, ref_id, out_file_path, all_out=None, mhc=None):
    """
    Print all combine metric in file like[HLA-B*58-01_compare_result.txt]
    """
    data = get_data_from_result_file(result_file)
    out_file = open(out_file_path, 'w')

    # print(data, '\n\n')

    title = 'data_type\tcount\tpos_count\tneg_count\t'
    for method_name in DATA_TEMPLATE_NEW_FORMAT_METHODS_LIST:
        title += '{method_name}_auc\t{method_name}_prcc\t{method_name}_pearson\t'.format(
            method_name=method_name)
    out_file.write(title+'\n')

    for measure_type in data:
        meausre_data = data[measure_type]

        if meausre_data['count'] == 0:
            continue

        out_str = '{type}'.format(type=measure_type)

        real = meausre_data['real']

        count, pos_count, neg_count = WEEK_DATA_COUNT(
            real, convert=measure_type)
        out_str += '\t{}\t{}\t{}'.format(count, pos_count, neg_count)
        print(mhc, measure_type, meausre_data['count'])
        for method in meausre_data['methods']:
            # only compare predict all values
            if len(meausre_data['methods'][method]) != len(real):
                out_str += '\t-\t-\t-'
                continue

            if 'ConvBinding' in method:
                auc_score = IEDB_AUC_binary_prediction(
                    real, meausre_data['methods'][method], convert=measure_type)
                prcc = IEDB_PRCC_binary_prediction(
                    real, meausre_data['methods'][method], convert=measure_type)
                pearsonr_score = stand_pearson_binary_prediction(
                    real, meausre_data['methods'][method], convert=measure_type)
            else:
                auc_score = IEDB_AUC(
                    real, meausre_data['methods'][method], convert=measure_type)
                prcc = IEDB_PRCC(
                    real, meausre_data['methods'][method], convert=measure_type)
                pearsonr_score = stand_pearson(
                    real, meausre_data['methods'][method], convert=measure_type)

            out_str += '\t{auc_score}\t{prcc}\t{pearsonr_score}'.format(
                auc_score=auc_score, prcc=prcc, pearsonr_score=pearsonr_score)

        out_file.write(out_str + '\n')

        if all_out != None:
            all_out.write(mhc + '\t' + ref_id + '\t' + out_str + '\n')
    out_file.close()


def print_out_metrics_file():
    # combine metricfile
    all_out_file_path = os.path.join(
        TEST_RESULT_DIR, 'Combine_result_compre_ALL.txt')
    all_out_file = open(all_out_file_path, 'w')

    # title
    title = 'mhc\tiedb_ref_id\tdata_type\tcount\tpos_count\tneg_count\t'
    for method_name in DATA_TEMPLATE_NEW_FORMAT_METHODS_LIST:
        title += '{method_name}_auc\t{method_name}_prcc\t{method_name}_pearson\t'.format(
            method_name=method_name)
    all_out_file.write(title+'\n')

    for result_file_name in RESULT_FILE_LIST:
        result_file_path = os.path.join(TEST_RESULT_DIR, result_file_name)

        mhc = result_file_name.split('_')[0]
        ref_id = result_file_name.split('_')[1]

        out_file_name = '{}_{}_compare_result.txt'.format(mhc, ref_id)
        out_file_path = os.path.join(TEST_RESULT_DIR, out_file_name)

        pass_allle_result(result_file_path, ref_id,
                          out_file_path, all_out=all_out_file, mhc=mhc)


def test():
    global MODEL_DIR, BEST_MODEL, TEST_RESULT_DIR, RESULT_FILE_LIST

    # reset
    MODEL_DIR = os.path.join(BASE_DIR,
                             '..',
                             'models',
                             'benchmark_evaluation',
                             'best_model.keras')
    BEST_MODEL = load_model(MODEL_DIR)

    # reset
    RESULT_FILE_LIST = []

    # result dir
    TEST_RESULT_DIR = os.path.join(
        BASE_DIR, 'weekly_result_nonredundant_sep_iedbid')
    if not os.path.exists(TEST_RESULT_DIR):
        os.mkdir(TEST_RESULT_DIR)

    # do prediction
    available_hla_list = valid_allele_list()
    weekly_allele_list = weekly_data_alleles_list_NEW_NONREDUNT()
    for hla_allele in weekly_allele_list:
        if hla_allele in available_hla_list:
            test_weekly_data_on_allele(hla_allele)

    # analyze
    print_out_metrics_file()


if __name__ == '__main__':
    pass
