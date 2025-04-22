import unittest
import random


from utils import zeros, ones, eye, random_matrix, diag


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


if __name__ == "__main__":
    unittest.main()
