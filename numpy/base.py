class Matrix:
    def __init__(self, data=None, shape=None, fill_value=0):
        """
        Basic Operations:
        - __init__: Initialize matrix with data or shape
        - __str__: String representation of matrix
        - __repr__: String representation (same as str)
        - __getitem__: Get element or submatrix by index
        - __setitem__: Set element value
        - copy: Create a copy of matrix

        Arithmetic Operations:
        - __add__, __radd__: Matrix addition
        - __sub__, __rsub__: Matrix subtraction
        - __mul__, __rmul__: Element-wise multiplication
        - __truediv__, __rtruediv__: Element-wise division
        - __neg__: Unary minus
        - __eq__: Matrix equality comparison

        Matrix Operations:
        - dot: Matrix multiplication
        - T: Matrix transpose
        - det: Matrix determinant
        - inverse: Matrix inverse
        - rank: Matrix rank
        - trace: Matrix trace
        - diag: Get matrix diagonal

        Statistical Operations:
        - mean: Calculate mean (by axis or overall)
        - apply: Apply function to each element
        - shape: Get matrix dimensions
        """
        if data is not None:
            if isinstance(data, Matrix):
                self.data = [row[:] for row in data.data]
                self.shape = data.shape
            else:
                self.data = [row[:] for row in data]
                self.shape = (len(data), len(data[0]) if data else 0)
        elif shape is not None:
            rows, cols = shape
            self.data = [[fill_value for _ in range(cols)] for _ in range(rows)]
            self.shape = shape
        else:
            self.data = []
            self.shape = (0, 0)

    def __str__(self):
        """Строковое представление матрицы"""
        if not self.data:
            return "[]"

        rows = []
        for row in self.data:
            rows.append(
                "["
                + " ".join(
                    f"{x:8.4f}" if isinstance(x, float) else f"{x:8}" for x in row
                )
                + "]"
            )

        return "\n".join(rows)

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, indices):
        """Получение элемента или подматрицы"""
        if isinstance(indices, tuple):
            i, j = indices
            if isinstance(i, int) and isinstance(j, int):
                return self.data[i][j]
            elif isinstance(i, slice) and isinstance(j, slice):
                i_start, i_stop, i_step = i.indices(self.shape[0])
                j_start, j_stop, j_step = j.indices(self.shape[1])

                result = []
                for row_idx in range(i_start, i_stop, i_step):
                    new_row = []
                    for col_idx in range(j_start, j_stop, j_step):
                        new_row.append(self.data[row_idx][col_idx])
                    result.append(new_row)

                return Matrix(result)
        elif isinstance(indices, int):
            return self.data[indices]

        raise IndexError("Неверный индекс")

    def __setitem__(self, indices, value):
        """Установка значения элемента матрицы"""
        if isinstance(indices, tuple):
            i, j = indices
            if isinstance(i, int) and isinstance(j, int):
                self.data[i][j] = value
            else:
                raise IndexError("Поддерживается только установка отдельных элементов")
        else:
            raise IndexError("Требуется кортеж индексов (строка, столбец)")

    def __add__(self, other):
        """Сложение матриц"""
        if isinstance(other, (int, float)):
            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self.data[i][j] + other
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сложить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self.data[i][j] + other.data[i][j]

        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Вычитание матриц"""
        if isinstance(other, (int, float)):
            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self.data[i][j] - other
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно вычесть матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self.data[i][j] - other.data[i][j]

        return result

    def __rsub__(self, other):
        """Вычитание справа"""
        if isinstance(other, (int, float)):
            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = other - self.data[i][j]
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно вычесть матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = other.data[i][j] - self.data[i][j]

        return result

    def __mul__(self, other):
        """Поэлементное умножение матриц или умножение на скаляр"""
        if isinstance(other, (int, float)):
            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self.data[i][j] * other
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно умножить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self.data[i][j] * other.data[i][j]

        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Поэлементное деление матриц или деление на скаляр"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")

            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self.data[i][j] / other
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно разделить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if other.data[i][j] == 0:
                    raise ZeroDivisionError("Деление на ноль")
                result[i, j] = self.data[i][j] / other.data[i][j]

        return result

    def __rtruediv__(self, other):
        """Деление справа"""
        if isinstance(other, (int, float)):
            result = Matrix(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.data[i][j] == 0:
                        raise ZeroDivisionError("Деление на ноль")
                    result[i, j] = other / self.data[i][j]
            return result

        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно разделить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.data[i][j] == 0:
                    raise ZeroDivisionError("Деление на ноль")
                result[i, j] = other.data[i][j] / self.data[i][j]

        return result

    def __neg__(self):
        """Унарный минус"""
        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = -self.data[i][j]
        return result

    def __eq__(self, other):
        """Сравнение матриц на равенство"""
        if not isinstance(other, Matrix):
            try:
                other = Matrix(other)
            except:
                return False

        if self.shape != other.shape:
            return False

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.data[i][j] != other.data[i][j]:
                    return False

        return True

    def __matmul__(self, other):
        """Матричное умножение"""
        if not isinstance(other, Matrix):
            other = Matrix(other)

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Невозможно выполнить матричное умножение: {self.shape} и {other.shape}"
            )

        result = Matrix(shape=(self.shape[0], other.shape[1]))

        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                sum_val = 0
                for k in range(self.shape[1]):
                    sum_val += self.data[i][k] * other.data[k][j]
                result[i, j] = sum_val

        return result

    @property
    def T(self):
        """Транспонирование матрицы"""
        result = Matrix(shape=(self.shape[1], self.shape[0]))

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[j, i] = self.data[i][j]

        return result

    def det(self):
        """Вычисление определителя матрицы"""
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                "Определитель можно вычислить только для квадратной матрицы"
            )

        n = self.shape[0]

        if n == 1:
            return self.data[0][0]

        if n == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]

        determinant = 0
        for j in range(n):
            # Создаем подматрицу, исключая первую строку и j-й столбец
            submatrix = []
            for i in range(1, n):
                row = []
                for k in range(n):
                    if k != j:
                        row.append(self.data[i][k])
                submatrix.append(row)

            # Рекурсивно вычисляем определитель подматрицы
            sign = (-1) ** j
            determinant += sign * self.data[0][j] * Matrix(submatrix).det()

        return determinant

    def inverse(self):
        """Вычисление обратной матрицы"""
        if self.shape[0] != self.shape[1]:
            raise ValueError(
                "Обратную матрицу можно вычислить только для квадратной матрицы"
            )

        det = self.det()
        if abs(det) < 1e-10:
            raise ValueError("Матрица вырожденная, обратной не существует")

        n = self.shape[0]

        if n == 1:
            return Matrix([[1 / self.data[0][0]]])

        # Матрица алгебраических дополнений
        cofactors = Matrix(shape=(n, n))

        for i in range(n):
            for j in range(n):
                # Создаем подматрицу, исключая i-ю строку и j-й столбец
                submatrix = []
                for r in range(n):
                    if r != i:
                        row = []
                        for c in range(n):
                            if c != j:
                                row.append(self.data[r][c])
                        submatrix.append(row)

                # Вычисляем минор и алгебраическое дополнение
                sign = (-1) ** (i + j)
                cofactors[i, j] = sign * Matrix(submatrix).det()

        # Транспонируем матрицу алгебраических дополнений и делим на определитель
        return cofactors.T * (1 / det)

    def rank(self):
        """Вычисление ранга матрицы"""
        # Создаем копию матрицы для приведения к ступенчатому виду
        m = Matrix(self)
        rows, cols = m.shape

        # Приведение к ступенчатому виду
        rank = 0
        row_used = [False] * rows

        for j in range(cols):
            for i in range(rows):
                if not row_used[i] and abs(m.data[i][j]) > 1e-10:
                    rank += 1
                    row_used[i] = True

                    # Нормализация строки
                    pivot = m.data[i][j]
                    for k in range(j, cols):
                        m.data[i][k] /= pivot

                    # Вычитание из других строк
                    for k in range(rows):
                        if k != i and abs(m.data[k][j]) > 1e-10:
                            factor = m.data[k][j]
                            for l in range(j, cols):
                                m.data[k][l] -= factor * m.data[i][l]

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

    def sum(self, axis=None):
        """Суммирование элементов матрицы"""
        if axis is None:
            # Сумма всех элементов
            total = 0
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    total += self.data[i][j]
            return total

        elif axis == 0:
            # Сумма по столбцам
            result = [0] * self.shape[1]
            for j in range(self.shape[1]):
                for i in range(self.shape[0]):
                    result[j] += self.data[i][j]
            return result

        elif axis == 1:
            # Сумма по строкам
            result = [0] * self.shape[0]
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i] += self.data[i][j]
            return result

        else:
            raise ValueError("Параметр axis должен быть None, 0 или 1")

    def mean(self, axis=None):
        """Среднее значение элементов матрицы"""
        if axis is None:
            # Среднее всех элементов
            return self.sum() / (self.shape[0] * self.shape[1])

        elif axis == 0:
            # Среднее по столбцам
            sums = self.sum(axis=0)
            return [s / self.shape[0] for s in sums]

        elif axis == 1:
            # Среднее по строкам
            sums = self.sum(axis=1)
            return [s / self.shape[1] for s in sums]

        else:
            raise ValueError("Параметр axis должен быть None, 0 или 1")

    def apply(self, func):
        """Применяет функцию к каждому элементу матрицы"""
        result = Matrix(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = func(self.data[i][j])
        return result

    def copy(self):
        """Создает копию матрицы"""
        return Matrix(self)

    def shape(self):
        """Возвращает размерность матрицы"""
        return self.shape


class array(Matrix):
    pass
