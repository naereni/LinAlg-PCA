class array:
    def __init__(self, data=None, shape=None, fill_value=0.0):
        """
        Basic Matrix Operations:
        - __init__: Create matrix from data or shape
        - __str__, __repr__: String representations
        - __getitem__: Access elements/submatrices
        - __setitem__: Modify elements
        - astype: Convert elements to new type
        - swap: Swap rows or columns
        - div: Divide row/column by scalar
        - comb: Add scaled row/column to another

        Arithmetic Operations:
        - __add__, __radd__: Addition (matrix + matrix, scalar + matrix)
        - __sub__, __rsub__: Subtraction (matrix - matrix, scalar - matrix)
        - __mul__, __rmul__: Element-wise multiplication
        - __truediv__, __rtruediv__: Element-wise division
        - __matmul__: Matrix multiplication
        - __neg__: Unary minus
        - __eq__: Equality comparison

        Linear Algebra Operations:
        - T: Transpose
        - det: Determinant
        - inv: Inverse matrix
        - rank: Matrix rank

        Properties:
        - shape: Matrix dimensions (rows, columns)
        """
        if data is not None:
            if isinstance(data, array):
                self.data = [list(row) for row in data.data]
                self.shape = data.shape
            else:
                if not isinstance(data, list):
                    raise TypeError("Входные данные 'data' должны быть списком.")

                if not data:
                    self.data = []
                    self.shape = (0, 0)

                else:
                    # Проверяем, является ли первый элемент списком (признак 2D структуры)
                    is_list_of_lists = isinstance(data[0], list)

                    if is_list_of_lists:
                        # Обработка списка списков (2D)
                        if not all(isinstance(row, list) for row in data):
                            raise TypeError(
                                "Входные данные 'data' должны быть списком списков."
                            )

                        rows = len(data)
                        cols = len(data[0])

                        processed_data = []
                        for i, row in enumerate(data):
                            if len(row) != cols:
                                raise ValueError(
                                    f"Несогласованная длина строк: строка {i} имеет длину {len(row)}, ожидалось {cols}."
                                )
                            processed_data.append([float(val) for val in row])

                        self.data = processed_data
                        self.shape = (rows, cols)
                    else:
                        # Обработка одномерного списка (1D) -> создаем матрицу-строку (1, n)
                        if any(isinstance(item, list) for item in data):
                            raise ValueError(
                                "Смешанные типы в одномерном списке не поддерживаются для создания матрицы."
                            )

                        rows = 1
                        cols = len(data)
                        self.data = [[float(val) for val in data]]
                        self.shape = (rows, cols)

        elif shape is not None:
            if not isinstance(shape, tuple) or len(shape) != 2:
                raise ValueError(
                    "Параметр 'shape' должен быть кортежем из двух целых чисел (rows, cols)."
                )
            rows, cols = shape
            if (
                not isinstance(rows, int)
                or not isinstance(cols, int)
                or rows < 0
                or cols < 0
            ):
                raise ValueError(
                    "Размеры матрицы (rows, cols) должны быть неотрицательными целыми числами."
                )

            self.data = [[float(fill_value) for _ in range(cols)] for _ in range(rows)]
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

    def __iter__(self):
        """
        Позволяет итерировать по матрице.
        Для обычных матриц итерация идет по строкам.
        Для матрицы-строки (1, n) или матрицы-столбца (n, 1) итерация идет по элементам.
        """
        rows, cols = self.shape

        if rows == 1:
            for j in range(cols):
                yield self.data[0][j]
        elif cols == 1:
            for i in range(rows):
                yield self.data[i][0]
        else:
            for i in range(rows):
                yield self.data[i]

    def __getitem__(self, indices):
        """Получение элемента или подматрицы"""
        if isinstance(indices, tuple):
            i, j = indices
            if isinstance(i, int) and isinstance(j, int):
                return self.data[i][j]
            elif isinstance(i, slice) and isinstance(j, int):
                i_start, i_stop, i_step = i.indices(self.shape[0])
                column = []
                for row_idx in range(i_start, i_stop, i_step):
                    column.append(
                        [self.data[row_idx][j]]
                    )  # Оборачиваем в список для создания столбца
                return array(column)
            elif isinstance(i, int) and isinstance(j, slice):
                j_start, j_stop, j_step = j.indices(self.shape[1])
                row = []
                for col_idx in range(j_start, j_stop, j_step):
                    row.append(self.data[i][col_idx])
                return array([row])  # Оборачиваем в список для создания строки
            elif isinstance(i, slice) and isinstance(j, slice):
                i_start, i_stop, i_step = i.indices(self.shape[0])
                j_start, j_stop, j_step = j.indices(self.shape[1])

                result = []
                for row_idx in range(i_start, i_stop, i_step):
                    new_row = []
                    for col_idx in range(j_start, j_stop, j_step):
                        new_row.append(self.data[row_idx][col_idx])
                    result.append(new_row)

                return array(result)
        elif isinstance(indices, int):
            return self.data[indices]

        raise IndexError("Неверный индекс")

    def __setitem__(self, indices, value):
        """Установка значения элемента матрицы"""
        if isinstance(indices, tuple):
            i, j = indices
            if isinstance(i, int) and isinstance(j, int):
                self.data[i][j] = float(value)
            elif isinstance(i, slice) and isinstance(j, int):
                i_start, i_stop, i_step = i.indices(self.shape[0])
                rows_count = (i_stop - i_start + i_step - 1) // i_step

                if isinstance(value, list):
                    if len(value) != rows_count:
                        raise ValueError(
                            f"Размер списка ({len(value)}) не соответствует количеству строк ({rows_count})"
                        )

                    idx = 0
                    for row_idx in range(i_start, i_stop, i_step):
                        self.data[row_idx][j] = float(value[idx])
                        idx += 1
                else:
                    for row_idx in range(i_start, i_stop, i_step):
                        self.data[row_idx][j] = float(value)
            elif isinstance(i, int) and isinstance(j, slice):
                j_start, j_stop, j_step = j.indices(self.shape[1])
                cols_count = (j_stop - j_start + j_step - 1) // i_step

                if isinstance(value, list):
                    if len(value) != cols_count:
                        raise ValueError(
                            f"Размер списка ({len(value)}) не соответствует количеству столбцов ({cols_count})"
                        )

                    idx = 0
                    for col_idx in range(j_start, j_stop, j_step):
                        self.data[i][col_idx] = float(value[idx])
                        idx += 1
                else:
                    for col_idx in range(j_start, j_stop, j_step):
                        self.data[i][col_idx] = float(value)
            else:
                raise IndexError(
                    "Поддерживается только установка отдельных элементов, строк или столбцов"
                )
        else:
            raise IndexError("Требуется кортеж индексов (строка, столбец)")

    def __add__(self, other):
        """Сложение матриц"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self[i, j] + other
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сложить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] + other[i, j]

        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Вычитание матриц"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self[i, j] - other
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно вычесть матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] - other[i, j]

        return result

    def __rsub__(self, other):
        """Вычитание справа"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = other - self.data[i][j]
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно вычесть матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = other.data[i][j] - self.data[i][j]

        return result

    def __mul__(self, other):
        """Поэлементное умножение матриц или умножение на скаляр"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self[i, j] * other
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно умножить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = self[i, j] * other[i, j]

        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """Поэлементное деление матриц или деление на скаляр"""
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Деление на ноль")

            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = self[i, j] / other
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно разделить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if other[i, j] == 0:
                    raise ZeroDivisionError("Деление на ноль")
                result[i, j] = self[i, j] / other[i, j]

        return result

    def __rtruediv__(self, other):
        """Деление справа"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.data[i][j] == 0:
                        raise ZeroDivisionError("Деление на ноль")
                    result[i, j] = other / self.data[i][j]
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно разделить поэлементно матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.data[i][j] == 0:
                    raise ZeroDivisionError("Деление на ноль")
                result[i, j] = other.data[i][j] / self.data[i][j]

        return result

    def __neg__(self):
        """Унарный минус"""
        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = -self.data[i][j]
        return result

    def __eq__(self, other):
        """Сравнение матриц на равенство"""
        if not isinstance(other, array):
            try:
                other = array(other)
            except:
                return False

        if self.shape != other.shape:
            return False

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self[i, j] != other[i, j]:
                    return False

        return True

    def __matmul__(self, other):
        """Матричное умножение"""
        if not isinstance(other, array):
            other = array(other)

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"Невозможно выполнить матричное умножение: {self.shape} и {other.shape}"
            )

        result = array(shape=(self.shape[0], other.shape[1]))

        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                sum_val = 0
                for k in range(self.shape[1]):
                    sum_val += self[i, k] * other[k, j]
                result[i, j] = sum_val

        return result

    def __lt__(self, other):
        """Поэлементное сравнение 'меньше чем' (self < other)"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = 1.0 if self[i, j] < other else 0.0
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сравнить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = 1.0 if self[i, j] < other[i, j] else 0.0

        return result

    def __gt__(self, other):
        """Поэлементное сравнение 'больше чем' (self > other)"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = 1.0 if self[i, j] > other else 0.0
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сравнить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = 1.0 if self[i, j] > other[i, j] else 0.0

        return result

    def __le__(self, other):
        """Поэлементное сравнение 'меньше или равно' (self <= other)"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = 1.0 if self[i, j] <= other else 0.0
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сравнить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = 1.0 if self[i, j] <= other[i, j] else 0.0

        return result

    def __ge__(self, other):
        """Поэлементное сравнение 'больше или равно' (self >= other)"""
        if isinstance(other, (int, float)):
            result = array(shape=self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result[i, j] = 1.0 if self[i, j] >= other else 0.0
            return result

        if not isinstance(other, array):
            other = array(other)

        if self.shape != other.shape:
            raise ValueError(
                f"Невозможно сравнить матрицы разных размеров: {self.shape} и {other.shape}"
            )

        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = 1.0 if self[i, j] >= other[i, j] else 0.0

        return result

    def swap(self, i, j, axis=0):
        """
        Меняет местами две строки (axis=0) или два столбца (axis=1).
        """
        rows, cols = self.shape
        if axis == 0:
            if i < 0 or i >= rows or j < 0 or j >= rows:
                raise IndexError("Индексы строк вне диапазона")
            if i == j:
                return self  # Нет смысла менять строку саму с собой
            self.data[i], self.data[j] = self.data[j], self.data[i]

        elif axis == 1:
            if i < 0 or i >= cols or j < 0 or j >= cols:
                raise IndexError("Индексы столбцов вне диапазона")
            if i == j:
                return self  # Нет смысла менять столбец сам с собой
            for row_idx in range(rows):
                self.data[row_idx][i], self.data[row_idx][j] = (
                    self.data[row_idx][j],
                    self.data[row_idx][i],
                )
        else:
            raise ValueError("Параметр 'axis' должен быть 0 (строки) или 1 (столбцы)")
        return self

    def div(self, index, k, axis=0):
        """
        Делит строку (axis=0) или столбец (axis=1) с индексом 'index' на скаляр 'k'.
        """
        rows, cols = self.shape
        if k == 0:
            raise ValueError("Деление на ноль невозможно")

        if axis == 0:  # Делим строку
            if index < 0 or index >= rows:
                raise IndexError("Индекс строки вне диапазона")
            for j in range(cols):
                self.data[index][j] /= k

        elif axis == 1:  # Делим столбец
            if index < 0 or index >= cols:
                raise IndexError("Индекс столбца вне диапазона")
            for i in range(rows):
                self.data[i][index] /= k
        else:
            raise ValueError("Параметр 'axis' должен быть 0 (строки) или 1 (столбцы)")
        return self

    def comb(self, target_idx, source_idx, k, axis=0):
        """
        Прибавляет к строке/столбцу target_idx строку/столбец source_idx, умноженную на k.
        axis=0: target_row += k * source_row
        axis=1: target_col += k * source_col
        """
        rows, cols = self.shape
        if axis == 0:  # Комбинируем строки
            if (
                target_idx < 0
                or target_idx >= rows
                or source_idx < 0
                or source_idx >= rows
            ):
                raise IndexError("Индексы строк вне диапазона")
            if target_idx == source_idx:
                return self
            for col in range(cols):
                self.data[target_idx][col] += k * self.data[source_idx][col]

        elif axis == 1:
            if (
                target_idx < 0
                or target_idx >= cols
                or source_idx < 0
                or source_idx >= cols
            ):
                raise IndexError("Индексы столбцов вне диапазона")
            if target_idx == source_idx:
                return self
            for row in range(rows):
                self.data[row][target_idx] += k * self.data[row][source_idx]
        else:
            raise ValueError("Параметр 'axis' должен быть 0 (строки) или 1 (столбцы)")
        return self

    # Оставляем старое имя для обратной совместимости
    def comb_rows(self, i, j, k):
        """Прибавляет k * строку j к строке i (для совместимости). Используйте comb_lines(i, j, k, axis=0)."""
        return self.comb(i, j, k, axis=0)

    def apply(self, func):
        """Применяет функцию к каждому элементу матрицы"""
        result = array(shape=self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result[i, j] = func(self[i, j])
        return result

    def copy(self):
        """Создает копию матрицы"""
        return array(self)


class array(array):
    pass
