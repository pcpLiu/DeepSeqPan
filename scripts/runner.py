from __future__ import absolute_import

import argparse

from train import train_2013
from eval import test

def train(*args, **kwargs):
    print("="*80)
    print("Train new model on [bdata.20130222.mhci.txt]")
    print("="*80)
    print("\n")

    train_2013()

    print("="*80)
    print("Finish Training...Model saved as best_model_2013.keras")
    print("="*80)
    print("\n")


def eval(*args, **kwargs):
    print("="*80)
    print("Eval model on [weekly_data_all_rm_duplicate.txt]")
    print("Model to eval: best_model.keras")
    print("="*80)
    print("\n")

    test()

    print("="*80)
    print("Finish Eval.")
    print("Check results in folder [weekly_result_nonredundant_sep_iedbid].")
    print("="*80)
    print("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Eval a model')
    parser.add_argument('--train', action='store_true', help='train a model')
    parser.add_argument('--eval', action='store_true', help='eval a model')
    args = parser.parse_args()

    if args.train:
        train()
        exit()

    if args.eval:
        eval()
        exit()
