import unittest

from base import Matrix


class TestMatrixArithmetic(unittest.TestCase):
    def test_add(self):
        """Тестирование сложения матриц"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        result = m1 + m2
        self.assertEqual(result.data, [[6, 8], [10, 12]])

        result = m1 + 2
        self.assertEqual(result.data, [[3, 4], [5, 6]])

        result = 2 + m1
        self.assertEqual(result.data, [[3, 4], [5, 6]])

        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m1 + m3

    def test_sub(self):
        """Тестирование вычитания матриц"""
        m1 = Matrix([[5, 6], [7, 8]])
        m2 = Matrix([[1, 2], [3, 4]])

        result = m1 - m2
        self.assertEqual(result.data, [[4, 4], [4, 4]])

        result = m1 - 2
        self.assertEqual(result.data, [[3, 4], [5, 6]])

        result = 10 - m2
        self.assertEqual(result.data, [[9, 8], [7, 6]])

        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m1 - m3

    def test_mul(self):
        """Тестирование поэлементного умножения матриц"""
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        result = m1 * m2
        self.assertEqual(result.data, [[5, 12], [21, 32]])

        result = m1 * 2
        self.assertEqual(result.data, [[2, 4], [6, 8]])

        result = 2 * m1
        self.assertEqual(result.data, [[2, 4], [6, 8]])

        m3 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m1 * m3

    def test_div(self):
        """Тестирование поэлементного деления матриц"""
        m1 = Matrix([[10, 20], [30, 40]])
        m2 = Matrix([[5, 4], [3, 8]])

        result = m1 / m2
        self.assertEqual(result.data, [[2, 5], [10, 5]])

        result = m1 / 2
        self.assertEqual(result.data, [[5, 10], [15, 20]])

        result = 60 / m2
        self.assertEqual(result.data, [[12, 15], [20, 7.5]])

        m3 = Matrix([[0, 1], [1, 0]])
        with self.assertRaises(ZeroDivisionError):
            m1 / m3

        m4 = Matrix([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m1 / m4

    def test_neg(self):
        """Тестирование унарного минуса"""
        m = Matrix([[1, -2], [-3, 4]])
        result = -m
        self.assertEqual(result.data, [[-1, 2], [3, -4]])

    def test_apply(self):
        """Тестирование применения функции к элементам матрицы"""
        m = Matrix([[1, 2], [3, 4]])

        result = m.apply(lambda x: x**2)
        self.assertEqual(result.data, [[1, 4], [9, 16]])

        result = m.apply(lambda x: x * 2)
        self.assertEqual(result.data, [[2, 4], [6, 8]])


if __name__ == "__main__":
    unittest.main()
