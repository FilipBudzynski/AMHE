from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from typing import Optional, Any
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# from scipy.optimize import minimize
import scipy
from sklearn.utils import optimize


class Surrogate(ABC):

    @abstractmethod
    def train(self):
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


class GussianProcessSurrogate(Surrogate):
    def __init__(
        self,
        min_data_to_train: Optional[float] = None,
        std_treshold: Optional[float] = None,
    ) -> None:
        self.x_real = []
        self.y_real = []
        self.model = GaussianProcessRegressor(
            # kernel=C(1.0) * RBF(length_scale=1.0),
            kernel=C(1.0, (1e-10, 1e10))
            * RBF(length_scale=1.0, length_scale_bounds=(1e-20, 1e6)),
            alpha=1e-6,
            normalize_y=True,
            optimizer=new_optimizer,
        )
        self.DEFAULT_DATA = 10
        self.min_data = min_data_to_train or self.DEFAULT_DATA
        self.std_treshold = 1e-6 or std_treshold

    def append(self, x, y):
        if isinstance(x, list) or len(np.array(x).shape) == 1:
            self.x_real.append(np.array(x))
            self.y_real.append(y)
        else:
            self.x_real.extend(x)
            self.y_real.extend(y)

        # Optional: prune old points to limit size
        MAX_POINTS = 1000
        if len(self.x_real) > MAX_POINTS:
            self.x_real = self.x_real[-MAX_POINTS:]
            self.y_real = self.y_real[-MAX_POINTS:]

    def train(self):
        if len(self.x_real) < self.min_data:
            return

        X_train = np.array(self.x_real)
        y = np.array(self.y_real)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)

        self.model.fit(X_scaled, y)

    def predict(self, pop):
        if len(self.x_real) >= self.min_data:
            X = np.array(pop)
            y_pred, std = self.model.predict(X, return_std=True)

            predictions = np.array(y_pred).flatten().astype(float).tolist()

            # Threshold-based selection: only evaluate uncertain points
            indices_to_evaluate = [
                i for i, s in enumerate(std) if s > self.std_treshold
            ]

            return indices_to_evaluate, predictions
        else:
            return list(range(len(pop))), [None] * len(pop)
