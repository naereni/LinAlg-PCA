import math
import nampy as np


def add_bias(X: np.array) -> np.array:
    n, m = X.shape
    ones = np.ones((n, 1))
    return np.hstack([ones, X])


class LinearRegression:
    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept
        self.w = None

    def fit(self, X: np.array, y: np.array) -> "LinearRegression":
        X_processed = X
        if self.add_intercept:
            X_processed = add_bias(X)

        # w = (X^T X)^(-1) X^T y
        X_T = X_processed.T
        self.w = (X_T @ X_processed).inv @ (X_T @ y)

        return self

    def predict(self, X: np.array) -> np.array:
        if self.w is None:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit().")

        X_processed = X
        if self.add_intercept:
            X_processed = add_bias(X)

        return X_processed @ self.w


def train_test_split(
    X: np.array, y: np.array, test_size: float = 0.2, random_seed: int = None
) -> tuple:
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Размерности не совпадают: X({X.shape}), y({y.shape})")

    if random_seed is not None:
        import random

        random.seed(random_seed)

    n = X.shape[0]
    test_count = int(n * test_size)

    # Создаем случайную перестановку индексов
    indices = list(range(n))
    import random

    random.shuffle(indices)

    # Разделяем индексы на обучающие и тестовые
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    # Создаем обучающую и тестовую выборки
    X_train = np.zeros((len(train_indices), X.shape[1]))
    X_test = np.zeros((len(test_indices), X.shape[1]))
    y_train = np.zeros((len(train_indices), 1))
    y_test = np.zeros((len(test_indices), 1))

    for i, idx in enumerate(train_indices):
        for j in range(X.shape[1]):
            X_train[i, j] = X[idx, j]
        y_train[i, 0] = y[idx, 0]

    for i, idx in enumerate(test_indices):
        for j in range(X.shape[1]):
            X_test[i, j] = X[idx, j]
        y_test[i, 0] = y[idx, 0]

    return X_train, X_test, y_train, y_test
