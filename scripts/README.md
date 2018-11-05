# Scripts

This folder contains code implementation of DeepSeqPan.
Also it has script to load model and do prediction.

- model.py: Defines the network structure of DeepSeqPan
- train.py: Training a new model
- eval.py: Evaluate a model
- model_arch.png: Model network arch
- model_input_output.txt: Input and output info of each layer

## Train

```bash
$ cd scripts
$ python runner --train
```

## Eval

```bash
$ cd scripts
$ python runner --eval
```

## Test
`python test.py [model_file] [testing_sample_file]`
```bash
$ cd scripts
$ python test.py ../models/benchmark_evaluation/best_model.keras sample_test.txt
```
The output will be like
```
HLA-B*27:05,HLNDETTSK,9.414250373840332 (log_ic50),0.06352277100086212 (binary)
HLA-B*27:05,GRQSTPRVS,6.252928733825684 (log_ic50),0.23112434148788452 (binary)
HLA-A*24:02,EYYFRNEVF,4.101010322570801 (log_ic50),0.5952587127685547 (binary)
```

## Env requirement

All codes and trained models were tested with **Python3.6** and following versioned packages:

- tensorflow == 1.4.0
- keras == 2.1.5