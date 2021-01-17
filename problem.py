import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from rampwf.score_types.base import BaseScoreType
import numpy as np

problem_title = "Prédiction du taux d'absentéisme"
_target_column_name = "% Abs/Ins"
_train = 'train.csv'
_test = 'test.csv'

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression(
    label_names=[])

# An object implementing the workflow
workflow = rw.workflows.Estimator()



class RMSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='RMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))


score_types = [
    RMSE(),
]

# Pour la cross validation
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=8, test_size=0.2, random_state=57)
    return cv.split(X, y)


# lecture des données
def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = _train
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = _test
    return _read_data(path, f_name)
