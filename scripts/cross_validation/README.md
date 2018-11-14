# Cross Validation codes
These codes are used to do cross validation tests.

## Usage

```python
# cv on cdhit filtered BD2013 data, 5 folds
python3 train.py CDHIT 5

# cv on cdhit raw BD2013, 5 folds, you can chose different folds
python3 train.py raw 5
```

Result are written into `cv_result.txt`