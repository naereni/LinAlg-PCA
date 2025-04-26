import unittest
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import array
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
        m1 = array([[1, 2], [3, 4]])
        m2 = array([[5, 6], [7, 8]])
        m3 = array([[9, 10]])
        m4 = array([[11], [12]])

        v_result = vstack([m1, m2])
        self.assertEqual(v_result.shape, (4, 2))
        self.assertEqual(v_result.data, [[1, 2], [3, 4], [5, 6], [7, 8]])

        v_result_concat = concat([m1, m2], axis=0)
        self.assertEqual(v_result_concat.data, [[1, 2], [3, 4], [5, 6], [7, 8]])

        h_result = hstack([m1, m4])
        self.assertEqual(h_result.shape, (2, 3))
        self.assertEqual(h_result.data, [[1, 2, 11], [3, 4, 12]])

        h_result_concat = concat([m1, m4], axis=1)
        self.assertEqual(h_result_concat.data, [[1, 2, 11], [3, 4, 12]])

        h_result_transposed = hstack([m1, m3.T])
        self.assertEqual(h_result_transposed.shape, (2, 3))
        self.assertEqual(h_result_transposed.data, [[1, 2, 9], [3, 4, 10]])

        with self.assertRaises(ValueError):
            concat([m1, m2], axis=2)

    def test_abs(self):
        """Тестирование абсолютных значений"""
        m = array([[-1, 2], [3, -4]])
        abs_m = abs(m)
        self.assertEqual(abs_m.data, [[1, 2], [3, 4]])

        m_zeros = zeros((2, 2))
        self.assertEqual(abs(m_zeros).data, [[0, 0], [0, 0]])

        with self.assertRaises(TypeError):
            abs([[1, 2], [3, 4]])

    def test_argmax(self):
        """Тестирование поиска индекса максимального элемента"""
        m = array([[1, 5, 3], [8, 2, 9]])

        self.assertEqual(argmax(m), 5)

        self.assertEqual(argmax(m, axis=0), [1, 0, 1])

        self.assertEqual(argmax(m, axis=1), [1, 2])

        m_neg = array([[-1, -5, -3], [-8, -2, -9]])
        self.assertEqual(argmax(m_neg), 0)
        self.assertEqual(argmax(m_neg, axis=0), [0, 1, 0])
        self.assertEqual(argmax(m_neg, axis=1), [0, 1])

        m_dup = array([[9, 5, 3], [8, 2, 9]])
        self.assertEqual(argmax(m_dup), 0)
        self.assertEqual(argmax(m_dup, axis=0), [0, 0, 1])
        self.assertEqual(argmax(m_dup, axis=1), [0, 2])

        m_empty = array(shape=(0, 0))
        with self.assertRaises(ValueError):
            argmax(m_empty)

        with self.assertRaises(ValueError):
            argmax(m, axis=2)

        with self.assertRaises(TypeError):
            argmax([[1, 2], [3, 4]])

    def test_sum(self):
        """Тестирование суммирования элементов"""
        m = array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(sum(m), 21)

        self.assertEqual(sum(m[:, 0]), 5)
        self.assertEqual(sum(m[:, 1]), 7)
        self.assertEqual(sum(m[:, 2]), 9)

        self.assertEqual(sum(m[0, :]), 6)
        self.assertEqual(sum(m[1, :]), 15)

        m_zeros = zeros((2, 2))
        self.assertEqual(sum(m_zeros), 0)
        self.assertEqual(sum(m_zeros[:, 0]), 0)
        self.assertEqual(sum(m_zeros[0, :]), 0)

    def test_mean(self):
        """Тестирование среднего значения элементов"""
        m = array([[1, 2, 3], [4, 5, 6]])

        self.assertAlmostEqual(mean(m), 21 / 6)

        self.assertAlmostEqual(mean(m[:, 0]), 5 / 2)
        self.assertAlmostEqual(mean(m[:, 1]), 7 / 2)
        self.assertAlmostEqual(mean(m[:, 2]), 9 / 2)

        self.assertAlmostEqual(mean(m[0, :]), 6 / 3)
        self.assertAlmostEqual(mean(m[1, :]), 15 / 3)

        m_zeros = zeros((2, 2))
        self.assertEqual(mean(m_zeros), 0)
        self.assertEqual(mean(m_zeros[:, 0]), 0)
        self.assertEqual(mean(m_zeros[0, :]), 0)

    def test_solve(self):
        """Тестирование решения СЛАУ"""
        A = array([[2, 1], [1, 3]])
        b = array([[5], [5]])
        x = solve(A, b)
        self.assertEqual(x.shape, (2, 1))
        self.assertAlmostEqual(x[0, 0], 2.0)
        self.assertAlmostEqual(x[1, 0], 1.0)

        check_b = A @ x
        self.assertAlmostEqual(check_b[0, 0], b[0, 0], delta=EPS)
        self.assertAlmostEqual(check_b[1, 0], b[1, 0], delta=EPS)

        A_list = [[2, 1], [1, 3]]
        b_list = [5, 5]
        x_list = solve(A_list, b_list)
        self.assertEqual(x_list.shape, (2, 1))
        self.assertAlmostEqual(x_list[0, 0], 2.0)
        self.assertAlmostEqual(x_list[1, 0], 1.0)

        A_singular = array([[1, 1], [1, 1]])
        b_singular = array([[2], [3]])
        with self.assertRaises(ValueError):
            solve(A_singular, b_singular)

        A_non_square = array([[1, 2, 3], [4, 5, 6]])
        b_non_square = array([[1], [2]])
        with self.assertRaises(ValueError):
            solve(A_non_square, b_non_square)

        A_dim_mismatch = array([[1, 2], [3, 4]])
        b_dim_mismatch = array([[1], [2], [3]])
        with self.assertRaises(ValueError):
            solve(A_dim_mismatch, b_dim_mismatch)


if __name__ == "__main__":
    unittest.main()
