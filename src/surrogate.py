from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from typing import Optional, Any


class Surrogate(ABC):

    @abstractmethod
    def train(self) -> bool:
        pass

    @abstractmethod
    def append(self, x, y) -> None:
        pass

    @abstractmethod
    def predict(self, pop) -> tuple[list[Any], list[Any]]:
        pass


class GussianProcessSurrogate(Surrogate):
    def __init__(self, min_data_to_train: Optional[float] = None) -> None:
        self.x_real = []
        self.y_real = []
        self.model = GaussianProcessRegressor(
            kernel=C(1.0) * RBF(length_scale=1.0), alpha=1e-6, normalize_y=True
        )
        self.DEFAULT_DATA = 20
        self.min_data = min_data_to_train or self.DEFAULT_DATA

    def append(self, x, y):
        self.x_real.append(x)
        self.y_real.append(y)

    def train(self):
        if len(self.x_real) < self.min_data:
            return False

        X = np.array(self.x_real)
        y = np.array(self.y_real)
        self.model.fit(X, y)
        return True

    def predict(self, pop):
        if hasattr(self.model, "predict") and len(self.x_real) >= 20:
            X = np.array(pop)
            y_pred = self.model.predict(X)

            predictions = np.array(y_pred).flatten().astype(float).tolist()

            N = int(0.3 * len(pop))
            top_indices = np.argsort(y_pred)[:N]

            return list(top_indices), predictions
        return [], [None] * len(pop)
