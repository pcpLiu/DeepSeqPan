import math


def metrics3():
    from sklearn.metrics import roc_curve, auc
    from scipy.stats import spearmanr

    all_results = []
    with open('cv_result.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue
            all_results.append(line.strip('\n').split(','))

    out_file = open('auc_srcc_of_ABC.txt', 'w')
    out_file.write('allele,auc_ic,srcc_ic,auc_binary,srcc_binary\n')
    for allele in ['-A','-B','-C']:
        real = []
        predict_ic = []
        predict_binary = []
        real_binary = []

        for info in filter(lambda x: allele in x[0], all_results):
            real.append(math.exp(float(info[-3])))
            real_binary = [1.0 if x < 500.0 else 0.0 for x in real]

            predict_ic.append(math.exp(float(info[-2])))
            predict_binary.append(float(info[-1]))

        print("="*25)
        print("IC50")
        fpr, tpr, _ = roc_curve(real_binary, [ -x for x in predict_ic])
        auc_ic = auc(fpr, tpr)
        srcc_ic = spearmanr(real, predict_ic)
        print('AUC:', auc_ic)
        print(srcc_ic)

        print("="*25)
        print("Binary")
        fpr, tpr, _ = roc_curve(real_binary, predict_binary)
        auc_binary = auc(fpr, tpr)
        srcc_binary = spearmanr(real, [-x for x in predict_binary])
        print('AUC:', auc_binary)
        print(srcc_binary)

        out_file.write('{},{},{},{},{}\n'.format(allele, auc_ic, srcc_ic[0],auc_binary,srcc_binary[0]))


def metrics2():
    from sklearn.metrics import roc_curve, auc
    from scipy.stats import spearmanr

    all_results = []
    with open('cv_result.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue
            all_results.append(line.strip('\n').split(','))

    alleles = alleles_list()
    out_file = open('auc_srcc_of_alleles.txt', 'w')
    out_file.write('allele,auc,srcc\n')
    for allele in alleles:
        real = []
        predict_ic = []
        predict_binary = []
        real_binary = []

        for info in filter(lambda x: x[0] == allele, all_results):
            real.append(math.exp(float(info[-3])))
            real_binary = [1.0 if x < 500.0 else 0.0 for x in real]

            predict_ic.append(math.exp(float(info[-2])))
            predict_binary.append(float(info[-1]))


        print("="*25)
        print("IC50")
        fpr, tpr, _ = roc_curve(real_binary, [ -x for x in predict_ic])
        if len(real_binary) >=2:
            try:
                auc_ic = auc(fpr, tpr)
            except:
                auc_ic = 'N/A'
        else:
            auc_ic = 'N/A'
        srcc_ic = spearmanr(real, predict_ic)
        print('AUC:', auc_ic)
        print(srcc_ic)

        print("="*25)
        print("Binary")
        fpr, tpr, _ = roc_curve(real_binary, predict_binary)
        if len(real_binary) >=2:
            try:
                auc_binary = auc(fpr, tpr)
            except:
                auc_binary = 'N/A'
        else:
            auc_binary = 'N/A'
        srcc_binary = spearmanr(real, [-x for x in predict_binary])
        print('AUC:', auc_binary)
        print(srcc_binary)

        out_file.write('{},{},{},{},{}\n'.format(allele, auc_ic, srcc_ic[0],auc_binary,srcc_binary[0]))

def metrics():
    from sklearn.metrics import roc_curve, auc
    from scipy.stats import spearmanr

    real = []
    predict_ic = []
    predict_binary = []
    real_binary = []
    with open('cv_result.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            info = line.strip('\n').split(',')
            real.append(math.exp(float(info[-3])))
            real_binary.append(1.0 if real[-1] < 500 else 0)
            predict_ic.append(math.exp(float(info[-2])))
            predict_binary.append(float(info[-1]))



    print("="*25)
    print("IC50")
    fpr, tpr, _ = roc_curve(real_binary, [ -x for x in predict_ic])
    print('AUC:', auc(fpr, tpr))
    print(spearmanr(real, predict_ic))

    print("="*25)
    print("Binary")
    fpr, tpr, _ = roc_curve(real_binary, predict_binary)
    print('AUC:', auc(fpr, tpr))
    print(spearmanr(real,  [ -x for x in predict_binary]))


def metrics4():
    from sklearn.metrics import roc_curve, auc
    # from scipy.stats import spearmanr
    from scipy.stats.mstats import spearmanr

    all_results = []
    with open('cv_result.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue
            all_results.append(line.strip('\n').split(','))

    alleles = ['HLA-A*02:10']
    for allele in alleles:
        predict = []
        real = []
        real_binary = []

        for info in filter(lambda x: x[0] == allele, all_results):
            real.append(math.exp(float(info[-2])))
            predict.append(math.exp(float(info[-1])))

            if math.exp(float(info[-2])) < 500:
                real_binary.append(1.0)
            else:
                real_binary.append(0.0)

        fpr, tpr, _ = roc_curve(real_binary, [ -x for x in predict])
        print('real,', real)
        print('predict,', predict)
        print('AUC:', auc(fpr, tpr))
        print(spearmanr(real, predict)[0])
        print('='*8)

def alleles_list():
    l = list()
    with open('alleles_list.txt', 'r') as in_file:
        for line in in_file:
            l.append(line.strip('\n'))
    return l

def list_allles():
    A = set()
    B = set()
    C = set()

    seq_count = [0, 0, 0]

    with open('cv_result.txt', 'r') as in_file:
        for line_num, line in enumerate(in_file):
            if line_num == 0:
                continue

            allele = line.strip('\n').split(',')[0]

            if 'A*' in allele:
                A.add(allele)
                seq_count[0] += 1
            elif 'B*' in allele:
                B.add(allele)
                seq_count[1] += 1
            else:
                C.add(allele)
                seq_count[2] += 1

    out_file = open('alleles_list.txt', 'w')
    print('A', len(A))
    print('B', len(B))
    print('C', len(C))
    for a in A:
        out_file.write(a + '\n')
    for a in B:
        out_file.write(a+ '\n')
    for a in C:
        out_file.write(a+ '\n')

    print(seq_count)

if __name__ == '__main__':
    # metrics()
    # list_allles()
    # metrics2()
    # metrics3()s
    # metrics4()
    metrics()
    pass