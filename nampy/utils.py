try:
    from .base import Matrix
except ImportError:
    from base import Matrix
import random
import math


EPS = 1e-6
INF = float("inf")
NINF = float("-inf")


def zeros(shape):
    """Создает матрицу, заполненную нулями"""
    return Matrix(shape=shape, fill_value=0)


def ones(shape):
    """Создает матрицу, заполненную единицами"""
    return Matrix(shape=shape, fill_value=1)


def eye(n):
    """Создает единичную матрицу размера n x n"""
    result = zeros((n, n))
    for i in range(n):
        result[i, i] = 1
    return result


def random_matrix(shape, min_val=0, max_val=1):
    """Создает матрицу со случайными значениями"""
    rows, cols = shape
    data = [
        [random.uniform(min_val, max_val) for _ in range(cols)] for _ in range(rows)
    ]
    return Matrix(data)


def diag(values):
    """Создает диагональную матрицу из списка значений"""
    n = len(values)
    result = zeros((n, n))
    for i in range(n):
        result[i, i] = values[i]
    return result


def solve(A, b):
    """Решает систему линейных уравнений A*x = b"""
    if not isinstance(A, Matrix):
        A = Matrix(A)

    if not isinstance(b, Matrix):
        if isinstance(b[0], (int, float)):
            b = Matrix([[val] for val in b])
        else:
            b = Matrix(b)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")

    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Размерности не совпадают: A {A.shape}, b {b.shape}")

    # Решаем систему через обратную матрицу
    return A.inv @ b


def concat(matrices, axis=0):
    """Объединяет матрицы"""
    if not matrices:
        return Matrix()

    if not all(isinstance(m, Matrix) for m in matrices):
        matrices = [Matrix(m) if not isinstance(m, Matrix) else m for m in matrices]

    if axis == 0:
        # Объединение по строкам (вертикально)
        if not all(m.shape[1] == matrices[0].shape[1] for m in matrices):
            raise ValueError("Все матрицы должны иметь одинаковое количество столбцов")

        result_data = []
        for matrix in matrices:
            result_data.extend([row for row in matrix.data])

        return Matrix(result_data)

    elif axis == 1:
        # Объединение по столбцам (горизонтально)
        if not all(m.shape[0] == matrices[0].shape[0] for m in matrices):
            raise ValueError("Все матрицы должны иметь одинаковое количество строк")

        result_data = [[] for _ in range(matrices[0].shape[0])]

        for i in range(matrices[0].shape[0]):
            for matrix in matrices:
                result_data[i].extend(matrix.data[i])

        return Matrix(result_data)

    else:
        raise ValueError("Параметр axis должен быть 0 или 1")


def vstack(matrices):
    """Вертикальное объединение матриц (по строкам)"""
    return concat(matrices, axis=0)


def hstack(matrices):
    """Горизонтальное объединение матриц (по столбцам)"""
    return concat(matrices, axis=1)


def abs(matrix):
    """Возвращает матрицу с абсолютными значениями элементов"""
    if not isinstance(matrix, Matrix):
        raise TypeError("Input must be a Matrix object")

    result_data = [[math.fabs(element) for element in row] for row in matrix.data]
    return Matrix(result_data)


def argmax(matrix, axis=None):
    """
    Возвращает индексы максимальных элементов вдоль заданной оси.

    Args:
        matrix (Matrix): Входная матрица.
        axis (int, optional): Ось, вдоль которой ищется максимум.
                              None: поиск по всей матрице (возвращает плоский индекс).
                              0: поиск по столбцам (возвращает индексы строк).
                              1: поиск по строкам (возвращает индексы столбцов).
                              Defaults to None.

    Returns:
        int or list: Индекс(ы) максимального элемента.
                     - int: если axis is None.
                     - list: список индексов строк (если axis=0) или столбцов (если axis=1).

    Raises:
        ValueError: Если матрица пуста или указана неверная ось.
    """
    if not isinstance(matrix, Matrix):
        raise TypeError("Input must be a Matrix object")

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise ValueError("Cannot perform argmax on an empty matrix")

    if axis is None:
        # Поиск по всей матрице (плоский индекс)
        max_val = -float("inf")
        max_idx = -1
        for r in range(rows):
            for c in range(cols):
                if matrix.data[r][c] > max_val:
                    max_val = matrix.data[r][c]
                    max_idx = r * cols + c  # Вычисляем плоский индекс
        return max_idx

    elif axis == 0:
        # Поиск максимума в каждом столбце (возвращает индекс строки)
        max_indices = []
        for c in range(cols):
            max_val_col = -float("inf")
            max_row_idx = -1
            for r in range(rows):
                if matrix.data[r][c] > max_val_col:
                    max_val_col = matrix.data[r][c]
                    max_row_idx = r
            max_indices.append(max_row_idx)
        # Можно вернуть как список или как Matrix(1, cols)
        # return Matrix([max_indices])
        return max_indices  # Возвращаем список для простоты

    elif axis == 1:
        # Поиск максимума в каждой строке (возвращает индекс столбца)
        max_indices = []
        for r in range(rows):
            max_val_row = -float("inf")
            max_col_idx = -1
            for c in range(cols):
                if matrix.data[r][c] > max_val_row:
                    max_val_row = matrix.data[r][c]
                    max_col_idx = c
            max_indices.append(max_col_idx)
        # Можно вернуть как список или как Matrix(rows, 1)
        # return Matrix([[idx] for idx in max_indices])
        return max_indices  # Возвращаем список для простоты

    else:
        raise ValueError("Параметр axis должен быть None, 0 или 1")


def sum(matrix, axis=None):
    """Суммирование элементов матрицы"""
    if axis is None:
        # Сумма всех элементов
        total = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                total += matrix.data[i][j]
        return total

    elif axis == 0:
        # Сумма по столбцам
        result = [0] * matrix.shape[1]
        for j in range(matrix.shape[1]):
            for i in range(matrix.shape[0]):
                result[j] += matrix.data[i][j]
        return result

    elif axis == 1:
        # Сумма по строкам
        result = [0] * matrix.shape[0]
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                result[i] += matrix.data[i][j]
        return result

    else:
        raise ValueError("Параметр axis должен быть None, 0 или 1")


def mean(matrix, axis=None):
    """Среднее значение элементов матрицы"""
    if axis is None:
        # Среднее всех элементов
        return sum(matrix) / (matrix.shape[0] * matrix.shape[1])

    elif axis == 0:
        # Среднее по столбцам
        sums = sum(matrix, axis=0)
        return [s / matrix.shape[0] for s in sums]

    elif axis == 1:
        # Среднее по строкам
        sums = sum(matrix, axis=1)
        return [s / matrix.shape[1] for s in sums]

    else:
        raise ValueError("Параметр axis должен быть None, 0 или 1")
