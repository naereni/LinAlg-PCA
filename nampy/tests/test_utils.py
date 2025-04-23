import unittest
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import Matrix
from utils import (
    zeros,
    ones,
    eye,
    random_matrix,
    diag,
    concat,
    vstack,
    hstack,
    abs,
    argmax,
    sum,
    mean,
    solve,
    EPS,
)


class TestUtils(unittest.TestCase):
    def test_zeros(self):
        """Тестирование создаия нулевой матрицы"""
        m = zeros((2, 3))
        self.assertEqual(m.shape, (2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(m[i, j], 0)

    def test_ones(self):
        """Тестирование создания матрицы из единиц"""
        m = ones((2, 3))
        self.assertEqual(m.shape, (2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(m[i, j], 1)

    def test_eye(self):
        """Тестирование создания единичной матрицы"""
        m = eye(3)
        self.assertEqual(m.shape, (3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertEqual(m[i, j], 1)
                else:
                    self.assertEqual(m[i, j], 0)

    def test_random_matrix(self):
        """Тестирование создания случайной матрицы"""

        random.seed(42)

        m = random_matrix((2, 3), min_val=0, max_val=1)
        self.assertEqual(m.shape, (2, 3))

        for i in range(2):
            for j in range(3):
                self.assertTrue(0 <= m[i, j] <= 1)

    def test_diag(self):
        """Тестирование создания диагональной матрицы"""
        m = diag([1, 2, 3])
        self.assertEqual(m.shape, (3, 3))

        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertEqual(m[i, j], i + 1)
                else:
                    self.assertEqual(m[i, j], 0)

    def test_concat_vstack_hstack(self):
        """Тестирование объединения матриц"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        m3 = Matrix([[9, 10]])
        m4 = Matrix([[11], [12]])

        # vstack (axis=0)
        v_result = vstack([m1, m2])
        self.assertEqual(v_result.shape, (4, 2))
        self.assertEqual(v_result.data, [[1, 2], [3, 4], [5, 6], [7, 8]])

        v_result_concat = concat([m1, m2], axis=0)
        self.assertEqual(v_result_concat.data, [[1, 2], [3, 4], [5, 6], [7, 8]])

        # hstack (axis=1)
        h_result = hstack([m1, m4])
        self.assertEqual(h_result.shape, (2, 3))
        self.assertEqual(h_result.data, [[1, 2, 11], [3, 4, 12]])

        h_result_concat = concat([m1, m4], axis=1)
        self.assertEqual(h_result_concat.data, [[1, 2, 11], [3, 4, 12]])

        # Проверка hstack с транспонированной матрицей (должно работать)
        h_result_transposed = hstack([m1, m3.T])
        self.assertEqual(h_result_transposed.shape, (2, 3))
        self.assertEqual(h_result_transposed.data, [[1, 2, 9], [3, 4, 10]])

        with self.assertRaises(ValueError):  # Неверная ось
            concat([m1, m2], axis=2)

    def test_abs(self):
        """Тестирование абсолютных значений"""
        m = Matrix([[-1, 2], [3, -4]])
        abs_m = abs(m)
        self.assertEqual(abs_m.data, [[1, 2], [3, 4]])

        m_zeros = zeros((2, 2))
        self.assertEqual(abs(m_zeros).data, [[0, 0], [0, 0]])

        with self.assertRaises(TypeError):
            abs([[1, 2], [3, 4]])  # Должен быть объект Matrix

    def test_argmax(self):
        """Тестирование поиска индекса максимального элемента"""
        m = Matrix([[1, 5, 3], [8, 2, 9]])

        # axis=None (плоский индекс)
        self.assertEqual(
            argmax(m), 5
        )  # 9 находится в (1, 2), плоский индекс 1*3 + 2 = 5

        # axis=0 (индексы строк)
        self.assertEqual(
            argmax(m, axis=0), [1, 0, 1]
        )  # Макс в столбцах: 8 (индекс 1), 5 (индекс 0), 9 (индекс 1)

        # axis=1 (индексы столбцов)
        self.assertEqual(
            argmax(m, axis=1), [1, 2]
        )  # Макс в строках: 5 (индекс 1), 9 (индекс 2)

        m_neg = Matrix([[-1, -5, -3], [-8, -2, -9]])
        self.assertEqual(argmax(m_neg), 0)  # -1 в (0,0)
        self.assertEqual(argmax(m_neg, axis=0), [0, 1, 0])
        self.assertEqual(argmax(m_neg, axis=1), [0, 1])

        m_dup = Matrix([[9, 5, 3], [8, 2, 9]])
        self.assertEqual(argmax(m_dup), 0)  # Первый 9 в (0,0)
        self.assertEqual(argmax(m_dup, axis=0), [0, 0, 1])
        self.assertEqual(argmax(m_dup, axis=1), [0, 2])

        m_empty = Matrix(shape=(0, 0))
        with self.assertRaises(ValueError):
            argmax(m_empty)

        with self.assertRaises(ValueError):  # Неверная ось
            argmax(m, axis=2)

        with self.assertRaises(TypeError):
            argmax([[1, 2], [3, 4]])  # Должен быть объект Matrix

    def test_sum(self):
        """Тестирование суммирования элементов"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        # axis=None
        self.assertEqual(sum(m), 21)

        # axis=0 (по столбцам)
        self.assertEqual(sum(m, axis=0), [5, 7, 9])

        # axis=1 (по строкам)
        self.assertEqual(sum(m, axis=1), [6, 15])

        m_zeros = zeros((2, 2))
        self.assertEqual(sum(m_zeros), 0)
        self.assertEqual(sum(m_zeros, axis=0), [0, 0])
        self.assertEqual(sum(m_zeros, axis=1), [0, 0])

        with self.assertRaises(ValueError):  # Неверная ось
            sum(m, axis=2)

    def test_mean(self):
        """Тестирование среднего значения элементов"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        # axis=None
        self.assertAlmostEqual(mean(m), 21 / 6)

        # axis=0 (по столбцам)
        mean_ax0 = mean(m, axis=0)
        self.assertAlmostEqual(mean_ax0[0], 5 / 2)
        self.assertAlmostEqual(mean_ax0[1], 7 / 2)
        self.assertAlmostEqual(mean_ax0[2], 9 / 2)

        # axis=1 (по строкам)
        mean_ax1 = mean(m, axis=1)
        self.assertAlmostEqual(mean_ax1[0], 6 / 3)
        self.assertAlmostEqual(mean_ax1[1], 15 / 3)

        m_zeros = zeros((2, 2))
        self.assertEqual(mean(m_zeros), 0)
        self.assertEqual(mean(m_zeros, axis=0), [0, 0])
        self.assertEqual(mean(m_zeros, axis=1), [0, 0])

        with self.assertRaises(ValueError):  # Неверная ось
            mean(m, axis=2)

    def test_solve(self):
        """Тестирование решения СЛАУ"""
        A = Matrix([[2, 1], [1, 3]])
        b = Matrix([[5], [5]])
        x = solve(A, b)
        self.assertEqual(x.shape, (2, 1))
        self.assertAlmostEqual(x[0, 0], 2.0)
        self.assertAlmostEqual(x[1, 0], 1.0)

        # Проверка A*x = b
        check_b = A @ x
        self.assertAlmostEqual(check_b[0, 0], b[0, 0], delta=EPS)
        self.assertAlmostEqual(check_b[1, 0], b[1, 0], delta=EPS)

        # Использование списков
        A_list = [[2, 1], [1, 3]]
        b_list = [5, 5]
        x_list = solve(A_list, b_list)
        self.assertEqual(x_list.shape, (2, 1))
        self.assertAlmostEqual(x_list[0, 0], 2.0)
        self.assertAlmostEqual(x_list[1, 0], 1.0)

        A_singular = Matrix([[1, 1], [1, 1]])
        b_singular = Matrix([[2], [3]])
        with self.assertRaises(ValueError):  # Вырожденная матрица
            solve(A_singular, b_singular)

        A_non_square = Matrix([[1, 2, 3], [4, 5, 6]])
        b_non_square = Matrix([[1], [2]])
        with self.assertRaises(ValueError):  # Не квадратная матрица A
            solve(A_non_square, b_non_square)

        A_dim_mismatch = Matrix([[1, 2], [3, 4]])
        b_dim_mismatch = Matrix([[1], [2], [3]])
        with self.assertRaises(ValueError):  # Несовпадение размерностей
            solve(A_dim_mismatch, b_dim_mismatch)


if __name__ == "__main__":
    unittest.main()
