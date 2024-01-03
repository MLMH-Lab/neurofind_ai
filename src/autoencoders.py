"""Autoencoder module."""
from pathlib import Path
import inspect
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas.core.generic import NDFrame
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from tensorflow import keras
from tqdm import tqdm

from src.utils import effect_size_score


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except IndexError:
    pass


def get_default_args(func):
    """Set default arguments.

    MLflow sends parametes in a way that prevents the functions to fill not given parameters with their default values.
    This function makes sure the all default values are settled.

    Parameters
    ----------
    func : callable
        Function for which we want to fill the default arguments.

    Returns
    -------
    dict
        Dictionary of parameters to feed the function required parameters.
    """
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


class dummy(BaseEstimator):
    """Test estimator."""

    def __init__(self, param_0=None, param_1=None):
        """Initialize mock model.

        Parameters
        ----------
        param_0 : str
            Dummy parameter.
        param_1 : type
            Dummy parameter..
        """
        self.param_0 = param_0
        self.param_1 = param_1

    def fit(self, X, y):
        """Fit dummy model.

        Parameters
        ----------
        X : array_like
            Any array like object.
        y : array_like
            Any array like object.
        """
        self.y = y
        return self

    def predict(self, X):
        """Predict dummy model.

        Parameters
        ----------
        X : array_like
            Dummy array.

        Returns
        -------
        array_like
            Same dummy array inputed.
        """
        return self.y[: X.shape[0]]


class Autoencoder(BaseEstimator):
    """Adversarial autoencoders approach.

    Parameters
    ----------
    pre_scaler_type : | 'RobustScaler' | 'StandardScaler' |, default : 'RobustScaler'
        Scaler to apply before the input of data to the autoencoder.
    pos_scaler_type : | 'RobustScaler' | 'StandardScaler' |, default : 'RobustScaler'
        Scaler to apply after the input of data to the autoencoder.
    base_learning_rate : float (0.0, 1.0), default : 0.0001
        The amount that the weights are updated during training.
        The learning rate controls how quickly the model is adapted to the problem. Smaller learning rates require more
        training epochs given the smaller changes made to the weights each update, whereas larger learning rates result
        in rapid changes and require fewer training epochs.
        A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution, whereas
        a learning rate that is too small can cause the process to get stuck.
    max_learning_rate : float, default : 0.005
    gamma : float, default : 0.98
    h_dim : list of integers
        List of the possible values numbers of neurons in each Dense network layers. One can use any number of layers as
        long as they are passed as a list. For instance h_dim = [10] will create one Dense layer with 10 neurons
        followed by a LeakyReLU layer, h_dim = [10, 100] will create 4 layers, one Dense layer with 10 neurons followed
        by a LeakyReLU layer and another Dense layer with a 100 neurons followed of a second LeakyReLU.
    z_dim : int, default : 20
        The number of neurons in the bottleneck layer of the Autoencoder.
    batch_size : int, default : 256
        The batch size limits the number of samples to be shown to the network before a weight update can be performed.
        This same limitation is then imposed when making predictions with the fit model.
    n_epochs : int, default : 200
        The number times that the learning algorithm will work through the entire training dataset.
        One epoch means that each sample in the training dataset has had an opportunity to update the internal model
        parameters.
    """

    def __init__(
        self,
        pre_scaler_type="RobustScaler",
        pos_scaler_type="RobustScaler",
        criterion='effect_size',
        base_learning_rate=0.0001,
        max_learning_rate=0.005,
        gamma=0.98,
        h_dim=[100, 100],
        z_dim=20,
        batch_size=256,
        n_epochs=11,
        verbose=3,
    ):
        """Initialize Autoencoder model.

        Parameters
        ----------
        pre_scaler_type : | 'RobustScaler' | 'StandardScaler' |, default : 'RobustScaler'
            Scaler to apply before the input of data to the autoencoder.
        pos_scaler_type : | 'RobustScaler' | 'StandardScaler' |, default : 'RobustScaler'
            Scaler to apply after the input of data to the autoencoder.
        base_learning_rate : float (0.0, 1.0), default : 0.0001
            The amount that the weights are updated during training.
            The learning rate controls how quickly the model is adapted to the problem. Smaller learning rates require
            more training epochs given the smaller changes made to the weights each update, whereas larger learning
            rates result in rapid changes and require fewer training epochs.
            A learning rate that is too large can cause the model to converge too quickly to a suboptimal solution,
            whereas a learning rate that is too small can cause the process to get stuck.
        max_learning_rate : float, default : 0.005
        gamma : float, default : 0.98
        h_dim : list of integers
            List of the possible values numbers of neurons in each Dense network layers. One can use any number of
            layers as long as they are passed as a list. For instance h_dim = [10] will create one Dense layer with 10
            neurons followed by a LeakyReLU layer, h_dim = [10, 100] will create 4 layers, one Dense layer with 10
            neurons followed by a LeakyReLU layer and another Dense layer with a 100 neurons followed of a second
            LeakyReLU.
        z_dim : int, default : 20
            The number of neurons in the bottleneck layer of the Autoencoder.
        batch_size : int, default : 256
            The batch size limits the number of samples to be shown to the network before a weight update can be
            performed. This same limitation is then imposed when making predictions with the fit model.
        n_epochs : int, default : 200
            The number times that the learning algorithm will work through the entire training dataset.
            One epoch means that each sample in the training dataset has had an opportunity to update the internal model
            parameters.
        """
        self.reset_none_params(locals())
        self.pre_scaler_type = pre_scaler_type
        self.pos_scaler_type = pos_scaler_type
        self.criterion = criterion
        self.base_learning_rate = base_learning_rate
        self.max_learning_rate = max_learning_rate
        self.gamma = gamma
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose

        # Pre-processing
        self.pre_scaler = self._create_pre_scaler(self.pre_scaler_type)
        self.enc_age = OneHotEncoder(sparse=False, handle_unknown="ignore")
        self.enc_gender = OneHotEncoder(sparse=False, handle_unknown="ignore")

        # Pos-processing
        self.pos_scaler = self._create_pos_scaler(self.pos_scaler_type)

    def _create_model(self):
        """Create all the components of the model."""
        self.enc = self._make_encoder(self.n_features, self.h_dim, self.z_dim)
        if self.verbose == 1:
            print(self.enc.summary())
        self.dec = self._make_decoder(self.n_features, self.h_dim, self.z_dim + self.n_labels)
        if self.verbose == 1:
            print(self.dec.summary())
        self.disc = self._make_discriminator(self.z_dim, self.h_dim)
        if self.verbose == 1:
            print(self.disc.summary())

        self.cross_entropy = keras.losses.BinaryCrossentropy(from_logits=True)
        self.mse = keras.losses.MeanSquaredError()
        self.acc = keras.metrics.BinaryAccuracy()

        self.ae_opt = tf.keras.optimizers.Adam(lr=self.base_learning_rate)
        self.dc_opt = tf.keras.optimizers.Adam(lr=self.base_learning_rate)
        self.gen_opt = tf.keras.optimizers.Adam(lr=self.base_learning_rate)

    def reset_none_params(self, parameters):
        """Reset the default parameters if any of the parameters are None."""
        default_params = get_default_args(Autoencoder)
        for parameter, value in parameters.items():
            if value is None:
                setattr(self, parameter, default_params[parameter])
            else:
                setattr(self, parameter, value)

    def _make_encoder(self, n_features, h_dim, z_dim):
        """Create encoder model.

        Parameters
        ----------
        n_features: int
            Number of input features.
        h_dim: int
            Number of neurons in each of the pairs of Dense and LeakyReLU layers.
        z_dim: int
            Number of neurons in the bottleneck layer. Ideally this should be smaller than any values in the h_dim list.

        Returns
        -------
        model: keras model class
            Groups layers into an object with training and inference features.
        """
        inputs = keras.Input(shape=(n_features,))
        x = inputs
        for n_neurons in h_dim:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)

        encoded = keras.layers.Dense(z_dim)(x)
        model = keras.Model(inputs=inputs, outputs=encoded)
        return model

    def _make_decoder(self, n_features, h_dim, encoded_dim):
        """Create encoder model.

        Parameters
        ----------
        encoded_dim : int
            Number of neurons in the input layer of the decoder. It is currently the number of neurons in the bottleneck
            (z_dim) layer plus the n_labels value.
        n_features : int
            Number of input features.
        h_dim : int
            Number of neurons in each of the pairs of Dense and LeakyReLU layers.

        Returns
        -------
        model : keras model class
            Groups layers into an object with training and inference features.
        """
        encoded = keras.Input(shape=(encoded_dim,))
        x = encoded
        for n_neurons in h_dim[::-1]:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)

        reconstruction = keras.layers.Dense(n_features, activation="linear")(x)
        model = keras.Model(inputs=encoded, outputs=reconstruction)
        return model

    def _make_discriminator(self, z_dim, h_dim):
        z_features = keras.Input(shape=(z_dim,))
        x = z_features
        for n_neurons in h_dim:
            x = keras.layers.Dense(n_neurons)(x)
            x = keras.layers.LeakyReLU()(x)

        prediction = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=z_features, outputs=prediction)
        return model

    def _discriminator_loss(self, real_output, fake_output):
        loss_real = self.cross_entropy(tf.ones_like(real_output), real_output)
        loss_fake = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        return loss_fake + loss_real

    def _generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def _train_step(self, batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            enc_output = self.enc(batch_x, training=True)
            dec_output = self.dec(tf.concat([enc_output, batch_y], axis=1), training=True)

            ae_loss = self.mse(batch_x, dec_output)

        ae_grads = ae_tape.gradient(ae_loss, self.enc.trainable_variables + self.dec.trainable_variables)

        self.ae_opt.apply_gradients(zip(ae_grads, self.enc.trainable_variables + self.dec.trainable_variables))

        # ---------------------------------------------------------------------
        # Discriminator
        with tf.GradientTape() as dc_tape:
            real_dist = tf.random.normal([batch_x.shape[0], self.z_dim], mean=0.0, stddev=1.0)

            enc_output = self.enc(batch_x, training=True)

            dc_real = self.disc(real_dist, training=True)
            dc_fake = self.disc(enc_output, training=True)

            # Discriminator Loss
            dc_loss = self._discriminator_loss(dc_real, dc_fake)

            # Discriminator Acc
            dc_acc = self.acc(
                tf.concat([tf.ones_like(dc_real), tf.zeros_like(dc_fake)], axis=0),
                tf.concat([dc_real, dc_fake], axis=0),
            )

        dc_grads = dc_tape.gradient(dc_loss, self.disc.trainable_variables)

        self.dc_opt.apply_gradients(zip(dc_grads, self.disc.trainable_variables))

        # ---------------------------------------------------------------------
        # Generator (Encoder)
        with tf.GradientTape() as gen_tape:
            enc_output = self.enc(batch_x, training=True)
            dc_fake = self.disc(enc_output, training=True)

            # Generator loss
            gen_loss = self._generator_loss(dc_fake)

        gen_grads = gen_tape.gradient(gen_loss, self.enc.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grads, self.enc.trainable_variables))
        return ae_loss, dc_loss, dc_acc, gen_loss

    def scale_fn(self, x):
        """Scale FN.

        Parameters
        ----------
        x : array_like (n_samples, n_features)
            Array of relative volume values.

        Returns
        -------
        arrey_like
            Gamma value times the `x` array.
        """
        return self.gamma ** x

    def _calc_batch(self, batch, batch_x, batch_y, step_size):
        # The floor of the scalar x is the largest integer i, such that i <= x.
        cycle = np.floor(1 + batch / (2 * step_size))
        x_lr = np.abs(batch / step_size - 2 * cycle + 1)

        clr = self.base_learning_rate + (self.max_learning_rate - self.base_learning_rate) * max(
            0, 1 - x_lr
        ) * self.scale_fn(cycle)

        self.ae_opt.lr = clr
        self.dc_opt.lr = clr
        self.gen_opt.lr = clr

        ae_loss, dc_loss, dc_acc, gen_loss = self._train_step(batch_x, batch_y)

        self.ae_loss_avg(ae_loss)
        self.dc_loss_avg(dc_loss)
        self.dc_acc_avg(dc_acc)
        self.gen_loss_avg(gen_loss)

    def fit(self, X):
        """Fit Autoencoder model.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.
        """
        if isinstance(X, NDFrame):
            X = X.values

        x = X[:, :-2]
        age = X[:, -2:-1]
        gender = X[:, -1:]
        # ---------------------------------------------------------------------
        # Pre-processing
        # ---------------------------------------------------------------------
        x_norm = self.pre_scaler.fit_transform(x)
        age = np.clip(age.astype(int), a_max=85, a_min=-1)
        age_one_hot = self.enc_age.fit_transform(age.astype(int))
        gender_one_hot = self.enc_gender.fit_transform(gender.astype(int))
        y_data = np.concatenate((age_one_hot, gender_one_hot), axis=1)
        y_data = y_data.astype("float32")

        self.n_features = x_norm.shape[1]
        self.n_labels = y_data.shape[1]
        self._create_model()

        n_samples = x_norm.shape[0]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_norm, y_data))
        train_dataset = train_dataset.shuffle(buffer_size=n_samples)
        train_dataset = train_dataset.batch(self.batch_size)

        # The ceil of the scalar x is the smallest integer i, such that i >= x. np.ceil(0.2) = 1.
        step_size = 2 * np.ceil(n_samples / self.batch_size)
        self.ae_loss = []
        self.dc_acc = []
        self.dc_loss = []
        self.gen_loss = []
        # ---------------------------------------------------------------------
        # Processing
        # ---------------------------------------------------------------------
        n_batches = len(train_dataset)
        for epoch in range(self.n_epochs):
            start = time.time()

            self.ae_loss_avg = tf.metrics.Mean()
            self.dc_acc_avg = tf.metrics.Mean()
            self.dc_loss_avg = tf.metrics.Mean()
            self.gen_loss_avg = tf.metrics.Mean()

            for batch, (batch_x, batch_y) in enumerate(train_dataset, start=1 + epoch * n_batches):
                self._calc_batch(batch, batch_x, batch_y, step_size)
            epoch_time = time.time() - start
            self.ae_loss_avg = self.ae_loss_avg.result().numpy()
            self.dc_acc_avg = self.dc_acc_avg.result().numpy()
            self.dc_loss_avg = self.dc_loss_avg.result().numpy()
            self.gen_loss_avg = self.gen_loss_avg.result().numpy()

            if self.verbose == 1:
                print(
                    (
                        f"{epoch:4d}: TIME: {epoch_time:.2f} "
                        f"ETA: {(epoch_time * (self.n_epochs - epoch)):.2f} "
                        f"DC_LOSS: {self.dc_loss_avg:.4f} "
                        f"AE_LOSS: {self.ae_loss_avg:.4f} "
                        f"DC_ACC: {self.dc_acc_avg:.4f} "
                        f"GEN_LOSS: {self.gen_loss_avg:.4f}"
                    )
                )
            self.ae_loss.append(self.ae_loss_avg)
            self.dc_acc.append(self.dc_acc_avg)
            self.dc_loss.append(self.dc_loss_avg)
            self.gen_loss.append(self.gen_loss_avg)

        encoded = self.enc(x_norm, training=False)
        recons = self.dec(tf.concat([encoded, y_data], axis=1), training=False)

        # ---------------------------------------------------------------------
        # Pos-processing
        # ---------------------------------------------------------------------

        self.recons_error = np.mean((x_norm - recons) ** 2, axis=1)
        self.recons_error = self.pos_scaler.fit_transform(self.recons_error[:, None])
        return self

    def score(self, X, y, test_groups=None):
        """Get model score.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.
        y : array-like of shape(n_samples)
            Diagnose of the subjects.

        Attributes
        ----------
        score_train_ : NDFrame
            Silhouette and effect size scores.
        """
        deviations = self.predict(X)
        finite_filter = np.isfinite(deviations)
        deviations = deviations[finite_filter]
        if self.criterion == 'effect_size':
            y = y[finite_filter.squeeze()]
            if test_groups is not None:
                test_groups = test_groups[finite_filter.squeeze()]
            self.score_train_ = effect_size_score(deviations, y, groups=test_groups)
            try:
                self.score_train_["silhouette_score"] = silhouette_score(deviations[:, None], y)
            except ValueError:
                self.score_train_["silhouette_score"] = -1.0
            return self
        elif self.criterion == 'mse':
            if test_groups is None:
                self.score_train_ = {}
                self.score_train_['mse'] = np.nanmean(abs(deviations))
            else:
                self.score_train_ = {}
                self.score_train_['mse'] = np.nanmean(abs(deviations[test_groups.groups.str.contains('test').values]))

    def fit_transform(self, X):
        """Fit and apply transformation to the input data.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.

        Returns
        -------
        array-like of shape(n_samples, 1)
            Reconstruction deviations from the original array.
        """
        self.fit(X)
        return self.predict(X)

    def transform(self, X):
        """Transform to the input data.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.

        Returns
        -------
        array-like of shape(n_samples, 1)
            Reconstruction deviations from the original array.
        """
        return self.predict(X)

    def predict(self, X):
        """Transform to the input data.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.

        Returns
        -------
        array-like of shape(n_samples, 1)
            Reconstruction deviations from the original array.
        """
        if isinstance(X, NDFrame):
            X = X.values

        x = X[:, :-2]
        age = X[:, -2:-1]
        gender = X[:, -1:]

        # Pre-processing
        x_norm = self.pre_scaler.transform(x)
        age_one_hot = self.enc_age.transform(age.astype(int))
        gender_one_hot = self.enc_gender.transform(gender.astype(int))

        y_data = np.concatenate((age_one_hot, gender_one_hot), axis=1)
        y_data = y_data.astype("float32")

        # ---------------------------------------------------------------------
        # Processing
        encoded = self.enc(x_norm, training=False)
        recons = self.dec(tf.concat([encoded, y_data], axis=1), training=False)
        # ---------------------------------------------------------------------
        # Pos-processing
        self.recons_error_ = np.mean((x_norm - recons) ** 2, axis=1)
        self.recons_error_ = self.pos_scaler.transform(self.recons_error_[:, None])

        return self.recons_error_

    def predict_regional_deviations(self, X):
        """Predict deviations by region.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.

        Returns
        -------
        array-like of shape(n_samples, n_features)
            Deviations for each single region.
        """
        # Pre-processing
        if isinstance(X, NDFrame):
            X = X.values

        x = X[:, :-2]
        age = X[:, -2:-1]
        gender = X[:, -1:]
        x_norm = self.pre_scaler.transform(x)

        age_one_hot = self.enc_age.transform(age.astype(int))
        gender_one_hot = self.enc_gender.transform(gender)

        y_data = np.concatenate((age_one_hot, gender_one_hot), axis=1)
        y_data = y_data.astype("float32")

        # ---------------------------------------------------------------------
        # Processing
        encoded = self.enc(x_norm, training=False)
        recons = self.dec(tf.concat([encoded, y_data], axis=1), training=False)

        return tf.keras.backend.get_value((x_norm - recons) ** 2)

    def save(self, path):
        """Save Keras models and sklearn models to files.

        Parameters
        ----------
        path : type
            Directory path for saving the models.
        """
        (path / "keras").mkdir(exist_ok=True, parents=True)
        tf.keras.models.save_model(self.enc, path / "keras/enc.h5")
        tf.keras.models.save_model(self.dec, path / "keras/dec.h5")
        tf.keras.models.save_model(self.disc, path / "keras/disc.h5")
        joblib.dump(self.enc_age, path / "keras/enc_age.joblib")
        joblib.dump(self.enc_gender, path / "keras/enc_gender.joblib")
        joblib.dump(self.pre_scaler, path / "keras/pre_scaler.joblib")
        joblib.dump(self.pre_scaler, path / "keras/pre_scaler.joblib")
        joblib.dump(self.pos_scaler, path / "keras/pos_scaler.joblib")

    def load(self, path):
        """Load the Keras models.

        Parameters
        ----------
        path : str or Path
            Path to the Keras folder containing all joblib and h5 files.

        Returns
        -------
        type
            Description of returned object.

        Examples
        -------
        Examples should be written in doctest format, and
        should illustrate how to use the function/class.
        >>>

        """
        self.enc = tf.keras.models.load_model(path / "keras/enc.h5")
        self.dec = tf.keras.models.load_model(path / "keras/dec.h5")
        self.disc = tf.keras.models.load_model(path / "keras/disc.h5")
        self.enc_age = joblib.load(path / "keras/enc_age.joblib")
        self.enc_gender = joblib.load(path / "keras/enc_gender.joblib")
        self.pre_scaler = joblib.load(path / "keras/pre_scaler.joblib")
        self.pos_scaler = joblib.load(path / "keras/pos_scaler.joblib")

    def _create_pre_scaler(self, input_scaler_type):
        if input_scaler_type == "RobustScaler":
            return RobustScaler()

        elif input_scaler_type == "StandardScaler":
            return StandardScaler()

        else:
            raise ValueError(f"{input_scaler_type} is not recognized as a possible scaler.")

    def _create_pos_scaler(self, output_scaler_type):
        if output_scaler_type == "RobustScaler":
            return RobustScaler()

        elif output_scaler_type == "StandardScaler":
            return StandardScaler()

        else:
            raise ValueError(output_scaler_type)


class GridSearchAAE(BaseEstimator):
    """Performs grid search adversarial autoencoders.

    Parameters
    ----------
    param_grid: dict
        Dictionary of parameters for the Autoencoder class.

    Attributes
    ----------
    results_: NDFrame
        DataFrame with the scores for all the possible combinations of the parameters in the param_grid. The scores
        include the effect_size for each of the patient groups compared with the healthy controls and the
        silhouette_score the distribution of deviations.
    best_score_: NDFrame
        The list of scores for the best combination of parameters. The best score is considered to be the ones that
        leads to the highest values of the harmonic mean of the effect_sizes and the silhouette_score.
    best_params_: Autoencoder object
        Estimator that was chosen by the search, i.e. estimator which gave highest score. If refit is True the model
        will be fitted.
    """

    def __init__(self,
                 param_grid={'base_learning_rate': [0.0001],
                             'max_learning_rate': [0.005],
                             'gamma': [0.98],
                             'h_dim': [[101, 101, 101, 101]],
                             'z_dim': [30],
                             'batch_size': [128]},
                 criterion='effect_size',
                 train_epochs=10,
                 epochs=11,
                 max_iter=None,
                 random_state=0,
                 refit=False,
                 ):
        """Instaciate GridSearchAAE class."""
        self.param_grid = param_grid
        self.criterion = criterion
        self.estimator = Autoencoder
        self.refit = refit
        self.best_params_ = None
        self.train_epochs = train_epochs
        self.epochs = epochs
        self.max_iter = max_iter
        # self.random_state = random_state if random_state is not None else int(time.time())
        self.random_state = random_state
        param_grid_default = {'base_learning_rate': [0.0001],
                              'max_learning_rate': [0.005],
                              'gamma': [0.98],
                              'h_dim': [[101, 101, 101, 101]],
                              'z_dim': [30],
                              'batch_size': [128]}
        for param_name, param_list in param_grid_default.items():
            if param_name not in param_grid.keys():
                param_grid[param_name] = param_grid_default[param_name]

    def get_best_params(self):
        """Look into the results and save return the information about the best estimator."""
        if self.criterion == 'effect_size':
            norm_ef_hmean = self.results_["ef_hmean_score"] / np.nanmax(self.results_["ef_hmean_score"])
            mean_score = (norm_ef_hmean + self.results_["silhouette_score"]) / 2
            best_id = mean_score.idxmax(skipna=True)
            best_id = 0 if best_id is np.nan else best_id
            best_score = self.results_.iloc[best_id][self.results_.columns[self.results_.columns.str.contains("score")]]
            best_params = pd.Series(self.candidates_params[best_id])
        elif self.criterion == 'mse':
            best_id = self.results_["mse"].idxmax(skipna=True)
            best_id = 0 if best_id is np.nan else best_id
            best_score = self.results_.iloc[best_id][self.results_.columns[self.results_.columns.str.contains("score")]]
            best_params = pd.Series(self.candidates_params[best_id])
        return best_score, best_params

    def built_best_estimator(self):
        """Built estimator with the best parameters."""
        if self.best_params_ is not None:
            try:
                self.best_params_["n_epochs"] = self.epochs
            except AttributeError:
                print(self.best_params_)
            self.best_estimator_ = self.estimator(**self.best_params_)
        else:
            raise NotFittedError(
                "This %s instance was initialized "
                "with refit=False. %s is "
                "available only after refitting on the best "
                "parameters. You can refit an estimator "
                "manually using the ``best_params_`` "
                "attribute" % (type(self).__name__, "built_best_estimator")
            )

    def fit_best_estimator(self, X):
        """Create model with the best_params and fit the model.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.
        """
        self.best_estimator_.fit(X)

    def _check_y(self, y):
        """Check the integrety of y.

        Parameters
        ----------
        y : array-like of shape(n_samples)
            Diagnose of the subjects.
        """
        if isinstance(y, NDFrame):
            y = y.values
        y_shape = y.shape
        if len(y_shape) > 1:
            if y_shape[1] > 1:
                raise (f"y has the shape [{y_shape}] should be a 1D array.")
            if y_shape[1] == 1:
                y = y.squeeze()
        return y

    def fit_transform(self, X, y, test_groups=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.
        y : array-like of shape(n_samples)
            Diagnose of the subjects.
        test_groups : array-like of shape(n_samples)
            Groups to measure the score in the test set.
        """
        self.fit(X, y, test_groups)
        self.fit_best_estimator(X)
        return self.best_estimator_.transform(X)

    def transform(self, X):
        """Use trained model to transform the data in X.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.

        Returns
        -------
        1D array
            Array of deviations.
        """
        return self.best_estimator_.transform(X)

    def fit(self, X, y, test_groups=None):
        """Run fit with all sets of parameters.

        Parameters
        ----------
        X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender]]
            Training vector, where n_samples is the number of subjects and
            n_features is the number of features in the freesurferData plus age, gender, tiv and diagnose.
            In order to work the model needs to receive a array with the all the features in the freesurferData followed
            by the age and gender.
        y : array-like of shape(n_samples)
            Diagnose of the subjects.
        test_groups : array-like of shape(n_samples)
            Groups to measure the score in the test set.
        """
        self.param_grid_train = self.param_grid.copy()
        self.param_grid_train["n_epochs"] = [self.train_epochs]
        if self.max_iter is None:
            self.candidates_params = list(ParameterGrid(self.param_grid_train))
        else:
            self.candidates_params = list(ParameterSampler(
                # self.param_grid_train, self.max_iter, random_state=self.random_state
                self.param_grid_train, self.max_iter, random_state=0
            ))
        y = self._check_y(y)
        if isinstance(X, NDFrame):
            X = X.values
        if test_groups is None:
            # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        else:
            X_train = X[y == 1]
            if (len(X[y != 1]) == 0) and (self.criterion == 'effect_size'):
                raise ValueError("There is no patience to test the models.")
        all_candidate_params = []
        all_out = []
        n_param = len(self.candidates_params)
        print(f"Testing {n_param} different sets of parameter.")
        for i, candidate_params in tqdm(enumerate(self.candidates_params),
                                        desc="Autoencoder hyperparameter search",
                                        total=n_param
                                        ):
            aae = self.estimator(**candidate_params, criterion=self.criterion)
            aae.fit(X_train)
            if test_groups is None:
                aae.score(X_test, y_test)
            else:
                aae.score(X, y, test_groups=test_groups)
            values = [[value] for value in candidate_params.values()]
            keys = list(candidate_params.keys())
            all_candidate_params.append(pd.DataFrame(values, index=keys, columns=[i]).T)
            scores = [[value] for value in aae.score_train_.values()]
            diagns = list(aae.score_train_.keys())
            all_out.append(pd.DataFrame(scores, index=diagns, columns=[i]).T)
        self.results_ = pd.concat(all_candidate_params)
        self.results_ = pd.concat([self.results_, pd.concat(all_out)], axis=1)
        self.best_score_, self.best_params_ = self.get_best_params()
        self.built_best_estimator()
        if self.refit:
            self.fit_best_estimator(X_train)
        return self

    def _verify_path(self, save_path):
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(exists_ok=True, parents=True)
        self.save_path_ = save_path

    def _delete_unpickleble_objects(self):
        delattr(self.best_estimator_, "enc")
        delattr(self.best_estimator_, "dec")
        delattr(self.best_estimator_, "disc")
        delattr(self.best_estimator_, "cross_entropy")
        delattr(self.best_estimator_, "mse")
        delattr(self.best_estimator_, "acc")
        delattr(self.best_estimator_, "ae_opt")
        delattr(self.best_estimator_, "dc_opt")
        delattr(self.best_estimator_, "gen_opt")

    def save(self, save_path, model_name="gridsearch_aae.joblib"):
        """Save model to directory.

        Parameters
        ----------
        save_path : str or Path
            Path to the directory where you want to save the model.
        model_name : str
            File name for the model to be recorded.
        """
        self._verify_path(save_path)
        if not hasattr(self, "best_estimator_"):
            self.built_best_estimator()
        self.best_estimator_.save(self.save_path_)
        self._delete_unpickleble_objects()
        joblib.dump(self, self.save_path_ / model_name)

    def load(self):
        """Load keras models."""
        if not hasattr(self, "best_estimator_"):
            self.built_best_estimator()
        self.best_estimator_.load(self.save_path_)
