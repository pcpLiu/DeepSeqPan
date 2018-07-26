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

## Env requirement

All codes and trained models were tested with **Python3.6** and following versioned packages:

- tensorflow == 1.4.0
- keras == 2.1.5