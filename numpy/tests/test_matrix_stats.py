import unittest

from base import Matrix


class TestMatrixStats(unittest.TestCase):
    def test_sum(self):
        """Тестирование суммирования элементов матрицы"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(m.sum(), 21)

        self.assertEqual(m.sum(axis=0), [5, 7, 9])

        self.assertEqual(m.sum(axis=1), [6, 15])

        with self.assertRaises(ValueError):
            m.sum(axis=2)

    def test_mean(self):
        """Тестирование вычисления среднего значения элементов матрицы"""
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(m.mean(), 3.5)

        self.assertEqual(m.mean(axis=0), [2.5, 3.5, 4.5])

        self.assertEqual(m.mean(axis=1), [2, 5])

    def test_max_min(self):
        """Тестирование нахождения максимального и минимального значений"""
        m = Matrix([[1, 8, 3], [4, 2, 6]])

        self.assertEqual(m.max(), 8)
        self.assertEqual(m.min(), 1)

        self.assertEqual(m.max(axis=0), [4, 8, 6])
        self.assertEqual(m.min(axis=0), [1, 2, 3])

        self.assertEqual(m.max(axis=1), [8, 6])
        self.assertEqual(m.min(axis=1), [1, 2])


if __name__ == "__main__":
    unittest.main()
