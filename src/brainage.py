"""Classes for brain age prediction.

Methods include:
    - Defining type of scaler, regression model and search space for
    - Training the regression model using cross-validation (CV);
    - Hyperparamater tuning using gridsearch
    - Predicting brain age in new subjects
    - Loading scaler and model to apply to new subjects

This class is used in brain_age.py.
"""
from sklearn_rvm.em_rvm import EMRVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR
import joblib
import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except IndexError:
    pass


class BrainAge():
    """ Brain age approach·"""

    def __init__(self,
                 regressor_type='Linear regression',
                 scaler_type='RobustScaler',
                 param_grid=None,
                 ):

        self.regressor = self._create_regressor(regressor_type)
        self.scaler = self._create_scaler(scaler_type)
        if param_grid is None:
            self.param_grid = self._create_param_grid(regressor_type)
        else:
            self.param_grid = param_grid

    def fit(self, X, age):
        # Input scaler
        x_norm = self.scaler.fit_transform(X)

        # ---------------------------------------------------------------------
        # Regression
        if self.param_grid is not None:
            gridsearch = GridSearchCV(self.regressor,
                                      param_grid=self.param_grid,
                                      scoring='neg_mean_absolute_error',
                                      refit=True,
                                      cv=10,
                                      verbose=3,
                                      n_jobs=30)

            gridsearch.fit(x_norm, age)

            self.regressor = gridsearch.best_estimator_
        else:
            self.regressor.fit(x_norm, age)

        predicted = self.regressor.predict(x_norm)

        # ---------------------------------------------------------------------
        self.abs_diff = age - predicted.reshape(-1, 1)
        return self

    def predict(self, X):
        """Predict brain age per subject.

        Parameters
        ----------
        X : type
            Independent regression variables to predict age, e.g. ROIs.

        Returns
        -------
        predicted : type
            Predicted brain age per subject.
        """
        # Pre-processing
        x_norm = self.scaler.transform(X)

        # ---------------------------------------------------------------------
        # Regression
        predicted = self.regressor.predict(x_norm)

        return predicted

    def save(self, path):
        """Save scaler or regressor models.

        Parameters
        ----------
        path : str
            Path to directory where files will be saved.
        """
        joblib.dump(self.scaler, f'{path}/brainage_scaler.joblib')
        joblib.dump(self.regressor, f'{path}/brainage_regressor.joblib')

    def load(self, path):
        """Load scaler or regressor models.

        Parameters
        ----------
        path : str
            Path to directory where are saved.
        """
        self.scaler = joblib.load(f'{path}/scaler.joblib')
        self.regressor = joblib.load(f'{path}/regressor.joblib')

    def _create_scaler(self, input_scaler_type):
        if input_scaler_type == 'RobustScaler':
            return RobustScaler()

        elif input_scaler_type == 'StandardScaler':
            return StandardScaler()

        else:
            raise ValueError(input_scaler_type)

    def _create_regressor(self, regressor_type):
        if regressor_type == 'Support vector regression':
            return SVR()

        elif regressor_type == 'Random forest regression':
            return RandomForestRegressor()

        elif regressor_type == 'Elastic net':
            return ElasticNet()

        elif regressor_type == 'Linear regression':
            return LinearRegression()

        elif regressor_type == 'RVM':
            return EMRVR()
        else:
            raise ValueError(regressor_type)

    def _create_param_grid(self, regressor_type):
        if regressor_type == 'Support vector regression':
            return {
                'kernel': ['linear'],
                'epsilon': [0.1],
                'tol': [1e-3],
                'C': [2 ** -7, 2 ** -5, 2 ** -3, 2 ** -2, 2 ** -1, 2 ** 0, 2 ** 1]
            }

        elif regressor_type == 'Random forest regression':
            return {
                'n_estimators': [50, 150, 250],
                'max_features': [0.25, 0.5, 0.75, 1.0],
                'min_samples_split': [2, 4, 6]
            }

        elif regressor_type == 'Elastic net':
            return {
                'max_iter': [1, 5, 10],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                'l1_ratio': np.arange(0.0, 1.0, 0.1)
            }

        elif regressor_type == 'RVM':
            return dict(
                kernel=['linear'],
                degree=[3],
                gamma=['auto_deprecated'],
                coef0=[0.0],
                tol=[0.001],
                threshold_alpha=[1e9],
                beta_fixed=['not_fixed'],
                alpha_max=[1e9],
                init_alpha=[None],
                bias_used=[True],
                max_iter=[5000],
                compute_score=[False],
                epsilon=[1e-08],
                verbose=[False],
            )

        elif regressor_type == 'Linear regression':
            return None

        else:
            raise ValueError(regressor_type)
