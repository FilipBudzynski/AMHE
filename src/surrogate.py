from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from typing import Optional, Any
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import scipy
from sklearn.utils import optimize
from sklearn.linear_model import LinearRegression


class Surrogate(ABC):
    @abstractmethod
    def train(self, t):
        pass

    @abstractmethod
    def append(self, x, y):
        pass

    @abstractmethod
    def predict(self, pop) -> tuple[list[Any], list[Any]]:
        pass


def new_optimizer(obj_func, initial_theta, bounds):
    result = scipy.optimize.minimize(
        obj_func,
        initial_theta,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 10},
    )
    return result.x, result.fun


class GaussianProcessSurrogate(Surrogate):
    def __init__(
        self,
        min_data_to_train: Optional[float] = None,
        std_treshold: Optional[float] = None,
        train_window_size=None,
        how_often_to_train=10,
    ) -> None:
        self.is_trained = False
        self.x_real = []
        self.y_real = []
        self.model = GaussianProcessRegressor(
            # kernel=C(1.0) * RBF(length_scale=1.0),
            kernel=C(1.0, (1e-5, 1e5))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
            alpha=1e-6,
            normalize_y=True,
            optimizer=new_optimizer,
        )
        self.DEFAULT_MIN_DATA = 200
        self.min_data = min_data_to_train or self.DEFAULT_MIN_DATA
        self.std_treshold = std_treshold or 1e-6
        self.train_window = train_window_size or self.DEFAULT_MIN_DATA
        self.how_often_to_train = how_often_to_train
        self.is_trained

    def _remove_duplicates(self, X_train, y_train):
        X_array = np.array(X_train)
        y_array = np.array(y_train)

        _, unique_indices = np.unique(X_array, axis=0, return_index=True)

        sorted_indices = np.sort(unique_indices)

        return X_array[sorted_indices], y_array[sorted_indices]

    def append(self, x, y):
        if isinstance(x, list) or len(np.array(x).shape) == 1:
            self.x_real.append(np.array(x))
            self.y_real.append(y)
        else:
            self.x_real.extend(x)
            self.y_real.extend(y)

        if len(self.x_real) > self.train_window:
            self.x_real = self.x_real[-self.train_window :]
            self.y_real = self.y_real[-self.train_window :]

    def train(self, t):
        if t % self.how_often_to_train != 0:
            return

        if len(self.x_real) < self.min_data:
            return

        X_train, y = self._remove_duplicates(self.x_real, self.y_real)
        # X_train = np.array(self.x_real)
        # y = np.array(self.y_real)

        scaler = preprocessing.StandardScaler().fit(X_train)
        self.scaler = scaler
        X_scaled = scaler.transform(X_train)

        self.model.fit(X_scaled, y)
        self.is_trained = True
        # self.x_real = []
        # self.y_real = []

    def predict(self, pop):
        if self.is_trained:
            X = np.array(pop)
            X = np.array(pop)
            X_scaled = self.scaler.transform(X)
            y_pred, std = self.model.predict(X_scaled, return_std=True)

            predictions = np.array(y_pred).flatten().astype(float).tolist()

            # Threshold-based selection: only evaluate uncertain points
            high_uncertainty_indices = [
                i for i, s in enumerate(std) if s < self.std_treshold
            ]

            return high_uncertainty_indices, predictions
        else:
            return list(range(len(pop))), [None] * len(pop)
