from base import Matrix
import random


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
    return A.inverse().dot(b)


def concatenate(matrices, axis=0):
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
            result_data.extend([row[:] for row in matrix.data])

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
    return concatenate(matrices, axis=0)


def hstack(matrices):
    """Горизонтальное объединение матриц (по столбцам)"""
    return concatenate(matrices, axis=1)
