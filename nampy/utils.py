try:
    from .base import array
except ImportError:
    from base import array
import random
import math
import csv
from builtins import abs as builtin_abs
from builtins import sum as builtin_sum


EPS = 1e-10
INF = float("inf")
NINF = float("-inf")


def zeros(shape):
    """Создает матрицу, заполненную нулями"""
    return array(shape=shape, fill_value=0.0)


def ones(shape):
    """Создает матрицу, заполненную единицами"""
    return array(shape=shape, fill_value=1.0)


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
    return array(data)


def diag(values):
    """Создает диагональную матрицу из списка значений"""
    n = len(values)
    result = zeros((n, n))
    for i in range(n):
        result[i, i] = values[i]
    return result


def std_scale(matrix, axis=0):
    """
    Стандартизирует матрицу (вычитает среднее и делит на стандартное отклонение).

    Стандартизация преобразует данные так, чтобы они имели среднее значение 0
    и стандартное отклонение 1.

    Args:
        matrix (array): Входная матрица для стандартизации
        axis (int): Ось, вдоль которой выполняется стандартизация:
                    0 - по столбцам (по умолчанию)
                    1 - по строкам
                    None - по всей матрице

    Returns:
        array: Стандартизированная матрица

    Raises:
        TypeError: Если входные данные не являются объектом Matrix
        ValueError: Если стандартное отклонение равно 0
    """
    if not isinstance(matrix, array):
        raise TypeError("Входные данные должны быть объектом Matrix")

    rows, cols = matrix.shape
    result = zeros(matrix.shape)

    if axis is None:
        mean_val = mean(matrix)
        std_val = matrix.std

        if builtin_abs(std_val) < EPS:
            raise ValueError(
                "Стандартное отклонение равно нулю, стандартизация невозможна"
            )

        mean_matrix = array(shape=matrix.shape, fill_value=mean_val)
        return (matrix - mean_matrix) / std_val

    elif axis == 0:
        for j in range(cols):
            col = matrix[:, j]

            col_mean = sum(col) / rows

            col_std = col.std

            if builtin_abs(col_std) < EPS:
                raise ValueError(
                    f"Стандартное отклонение столбца {j} равно нулю, стандартизация невозможна"
                )

            for i in range(rows):
                result[i, j] = (matrix[i, j] - col_mean) / col_std

        return result

    elif axis == 1:
        for i in range(rows):
            row = matrix[i, :]

            row_mean = sum(row) / cols

            row_std = row.std

            if builtin_abs(row_std) < EPS:
                raise ValueError(
                    f"Стандартное отклонение строки {i} равно нулю, стандартизация невозможна"
                )

            for j in range(cols):
                result[i, j] = (matrix[i, j] - row_mean) / row_std

        return result

    else:
        raise ValueError("Параметр axis должен быть 0, 1 или None")


def solve(A, b):
    """Решает систему линейных уравнений A*x = b"""
    if not isinstance(A, array):
        A = array(A)

    if not isinstance(b, array):
        if isinstance(b[0], (int, float)):
            b = array([[val] for val in b])
        else:
            b = array(b)

    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица A должна быть квадратной")

    if A.shape[0] != b.shape[0]:
        raise ValueError(f"Размерности не совпадают: A {A.shape}, b {b.shape}")

    return A.inv @ b


def concat(matrices, axis=0):
    """Объединяет матрицы"""
    if not matrices:
        return array()

    if not all(isinstance(m, array) for m in matrices):
        matrices = [array(m) if not isinstance(m, array) else m for m in matrices]

    if axis == 0:
        if not all(m.shape[1] == matrices[0].shape[1] for m in matrices):
            raise ValueError("Все матрицы должны иметь одинаковое количество столбцов")

        result_data = []
        for matrix in matrices:
            result_data.extend([row for row in matrix.data])

        return array(result_data)

    elif axis == 1:
        if not all(m.shape[0] == matrices[0].shape[0] for m in matrices):
            raise ValueError("Все матрицы должны иметь одинаковое количество строк")

        result_data = [[] for _ in range(matrices[0].shape[0])]

        for i in range(matrices[0].shape[0]):
            for matrix in matrices:
                result_data[i].extend(matrix.data[i])

        return array(result_data)

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
    if not isinstance(matrix, array):
        raise TypeError("Input must be a Matrix object")

    result_data = [[math.fabs(element) for element in row] for row in matrix.data]
    return array(result_data)


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
    if not isinstance(matrix, array):
        raise TypeError("Input must be a Matrix object")

    rows, cols = matrix.shape
    if rows == 0 or cols == 0:
        raise ValueError("Cannot perform argmax on an empty matrix")

    if axis is None:
        max_val = -float("inf")
        max_idx = -1
        for r in range(rows):
            for c in range(cols):
                if matrix.data[r][c] > max_val:
                    max_val = matrix.data[r][c]
                    max_idx = r * cols + c
        return max_idx

    elif axis == 0:
        max_indices = []
        for c in range(cols):
            max_val_col = -float("inf")
            max_row_idx = -1
            for r in range(rows):
                if matrix.data[r][c] > max_val_col:
                    max_val_col = matrix.data[r][c]
                    max_row_idx = r
            max_indices.append(max_row_idx)

        return max_indices

    elif axis == 1:
        max_indices = []
        for r in range(rows):
            max_val_row = -float("inf")
            max_col_idx = -1
            for c in range(cols):
                if matrix.data[r][c] > max_val_row:
                    max_val_row = matrix.data[r][c]
                    max_col_idx = c
            max_indices.append(max_col_idx)

        return max_indices

    else:
        raise ValueError("Параметр axis должен быть None, 0 или 1")


def sum(matrix):
    if not isinstance(matrix, array):
        raise TypeError("Входные данные должны быть объектом Matrix")

    total = 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            total += matrix.data[i][j]
    return total


def mean(matrix):
    return sum(matrix) / (matrix.shape[0] * matrix.shape[1])


def clip(x):
    """
    Clips values that are very close to integers within EPS tolerance.
    For example, 0.99999 becomes 1.0 if the difference is less than EPS.
    """
    if isinstance(x, (int, float)):
        nearest_int = round(x)
        if math.fabs(x - nearest_int) < EPS:
            return nearest_int
        return x
    elif isinstance(x, array):
        diff = abs(array([[round(val) - val for val in row] for row in x.data]))
        result_data = [
            [
                round(x.data[i][j]) if diff.data[i][j] < EPS else x.data[i][j]
                for j in range(x.shape[1])
            ]
            for i in range(x.shape[0])
        ]
        return array(result_data)
    else:
        raise TypeError("Input must be a number or Matrix object")


def all(arr):
    """
    Проверяет, являются ли все элементы массива истинными.

    Поддерживает работу с результатами сравнений, например:
    all(array < 5) проверит, все ли элементы array меньше 5.

    Args:
        arr: Матрица, список, кортеж или результат сравнения

    Returns:
        bool: True, если все элементы истинны, иначе False
    """
    if hasattr(arr, "__bool__"):
        try:
            print(arr)
            return bool(arr)
        except ValueError:
            pass

    if isinstance(arr, array):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if not arr[i, j]:
                    return False
        return True
    elif isinstance(arr, (list, tuple)):
        for item in arr:
            if not item:
                return False
        return True
    else:
        return bool(arr)


def read_csv(filename, target_col=-1, delimiter=",", dtype=float, index_col=None):
    """
    Читает данные из CSV файла и возвращает их в виде матрицы nampy

    Аргументы:
        filename (str): Путь к CSV файлу
        target_col (int): Индекс целевого столбца (по умолчанию -1)
        delimiter (str): Разделитель полей в CSV файле (по умолчанию ',')
        has_header (bool): Есть ли заголовок в файле (по умолчанию True)
        dtype: Тип данных для преобразования (по умолчанию float)
        index_col (int): Номер столбца с индексами (по умолчанию None)

    Возвращает:
        tuple: (X, y) где:
            X (Matrix): Матрица признаков
            y (Matrix): Вектор целевой переменной
    """
    with open(filename, "r", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        next(reader)

        feature_rows = []
        target_values = []

        for row in reader:
            if not row:
                continue

            try:
                data_row = row
                if index_col is not None:
                    data_row = row[:index_col] + row[index_col + 1 :]

                numeric_row = [dtype(val) if val.strip() else 0.0 for val in data_row]

                if target_col == -1:
                    target_col = len(numeric_row) - 1

                features = numeric_row[:target_col] + numeric_row[target_col + 1 :]
                target = numeric_row[target_col]

                feature_rows.append(features)
                target_values.append([target])

            except ValueError as e:
                print(e)

        if feature_rows:
            row_length = len(feature_rows[0])
            for i, row in enumerate(feature_rows):
                if len(row) != row_length:
                    raise ValueError(
                        f"Строка {i + 1} имеет неправильную длину: {len(row)} вместо {row_length}"
                    )

        return array(feature_rows), array(target_values)


def log_range(start, stop, num):
    log_start = math.log(start) if start > 0 else 0
    log_stop = math.log(stop)

    log_values = [
        log_start + i * (log_stop - log_start) / (num - 1) for i in range(num)
    ]
    values = [math.exp(x) for x in log_values]

    return values
