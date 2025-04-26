import math
from .base import array


def rmse(y_true: array, y_pred: array) -> float:
    """
    Вычисляет среднеквадратичную ошибку

    Вход:
        y_true: истинные значения (n*1)
        y_pred: предсказанные значения (n*1)

    Выход:
        mse: среднеквадратичная ошибка
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Размерности не совпадают: y_true({y_true.shape}), y_pred({y_pred.shape})"
        )

    n = y_true.shape[0]
    diff = y_true - y_pred
    squared_diff = diff * diff
    sum_squared_diff = 0

    for i in range(n):
        sum_squared_diff += squared_diff[i, 0]

    return math.sqrt(sum_squared_diff / n)
