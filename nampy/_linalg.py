from .base import array


def T(self):
    """Транспонирование матрицы"""
    result = array(shape=(self.shape[1], self.shape[0]))

    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            result[j, i] = self.data[i][j]

    return result


def det(self):
    """Вычисление определителя матрицы методом LU-разложения"""
    if self.shape[0] != self.shape[1]:
        raise ValueError("Определитель можно вычислить только для квадратной матрицы")

    n = self.shape[0]

    if n == 1:
        return self.data[0][0]

    if n == 2:
        return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

    # Создаем копию матрицы, чтобы не изменять оригинал
    A = array(self)

    # Счетчик перестановок строк для определения знака определителя
    sign_changes = 0

    # Массив для хранения информации о перестановках
    perm = list(range(n))

    # LU-разложение с выбором главного элемента
    for k in range(n - 1):
        # Поиск максимального элемента в текущем столбце
        max_row = k
        max_val = abs(A[k, k])

        for i in range(k + 1, n):
            if abs(A[i, k]) > max_val:
                max_val = abs(A[i, k])
                max_row = i

        # Если максимальный элемент равен нулю, определитель равен нулю
        if abs(max_val) < 1e-10:
            return 0.0

        # Если нужно, меняем строки местами
        if max_row != k:
            A.data[k], A.data[max_row] = A.data[max_row], A.data[k]
            perm[k], perm[max_row] = perm[max_row], perm[k]
            sign_changes += 1

        # Вычисляем множители для элементов под диагональю
        for i in range(k + 1, n):
            A[i, k] = A[i, k] / A[k, k]

            # Обновляем подматрицу
            for j in range(k + 1, n):
                A[i, j] -= A[i, k] * A[k, j]

    # Вычисляем определитель как произведение диагональных элементов
    determinant = 1.0
    for i in range(n):
        determinant *= A[i, i]

    # Учитываем изменения знака из-за перестановок строк
    if sign_changes % 2 == 1:
        determinant = -determinant

    return determinant


def inv(self):
    """Вычисление обратной матрицы методом Гаусса-Жордана"""
    if self.shape[0] != self.shape[1]:
        raise ValueError(
            "Обратную матрицу можно вычислить только для квадратной матрицы"
        )

    n = self.shape[0]

    if n == 1:
        if abs(self[0, 0]) < 1e-10:
            raise ValueError("Матрица вырожденная, обратной не существует")
        return array([[1 / self[0, 0]]])

    augmented = array(shape=(n, 2 * n))

    for i in range(n):
        for j in range(n):
            augmented[i, j] = self[i, j]

    for i in range(n):
        augmented[i, i + n] = 1.0

    for i in range(n):
        max_row = i
        max_val = abs(augmented[i, i])

        for k in range(i + 1, n):
            if abs(augmented[k, i]) > max_val:
                max_val = abs(augmented[k, i])
                max_row = k

        if max_val < 1e-10:
            raise ValueError("Матрица вырожденная, обратной не существует")

        if max_row != i:
            for j in range(2 * n):
                augmented[i, j], augmented[max_row, j] = (
                    augmented[max_row, j],
                    augmented[i, j],
                )

        pivot = augmented[i, i]
        for j in range(i, 2 * n):
            augmented[i, j] /= pivot

        for k in range(n):
            if k != i:
                factor = augmented[k, i]
                for j in range(i, 2 * n):
                    augmented[k, j] -= factor * augmented[i, j]

    inverse = array(shape=(n, n))
    for i in range(n):
        for j in range(n):
            inverse[i, j] = augmented[i, j + n]

    return inverse


def rank(self):
    """Вычисление ранга матрицы"""

    m = array(self)
    rows, cols = m.shape

    rank = 0
    row_used = [False] * rows

    for j in range(cols):
        for i in range(rows):
            if not row_used[i] and abs(m[i, j]) > 1e-10:
                rank += 1
                row_used[i] = True

                pivot = m[i, j]
                for k in range(j, cols):
                    m[i, k] /= pivot

                for k in range(rows):
                    if k != i and abs(m[k, j]) > 1e-10:
                        factor = m[k, j]
                        for l in range(j, cols):
                            m[k, l] -= factor * m[i, l]
                break
    return rank


def trace(self):
    """Вычисление следа матрицы"""
    if self.shape[0] != self.shape[1]:
        raise ValueError("След можно вычислить только для квадратной матрицы")

    trace_sum = 0
    for i in range(self.shape[0]):
        trace_sum += self.data[i][i]

    return trace_sum


def diag(self):
    """Возвращает диагональ матрицы в виде списка"""
    min_dim = min(self.shape)
    return [self.data[i][i] for i in range(min_dim)]


def norm(self):
    """Вычисление нормы матрицы"""
    sum_squares = 0.0
    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            sum_squares += self.data[i][j] ** 2

    return sum_squares**0.5


def std(self):
    """Вычисление стандартного отклонения элементов матрицы"""
    # Вычисляем среднее значение всех элементов
    sum_values = 0.0
    count = self.shape[0] * self.shape[1]

    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            sum_values += self.data[i][j]

    mean = sum_values / count

    # Вычисляем сумму квадратов отклонений от среднего
    sum_squared_diff = 0.0
    for i in range(self.shape[0]):
        for j in range(self.shape[1]):
            diff = self.data[i][j] - mean
            sum_squared_diff += diff * diff

    # Вычисляем стандартное отклонение
    return (sum_squared_diff / count) ** 0.5
