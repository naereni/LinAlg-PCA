import unittest


from base import array


class TestMatrixLinAlg(unittest.TestCase):
    def test_dot(self):
        """Тестирование матричного умножения"""
        m1 = array([[1, 2, 3], [4, 5, 6]])
        m2 = array([[7, 8], [9, 10], [11, 12]])

        result = m1 @ m2
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.data, [[58, 64], [139, 154]])

        m3 = array([[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            m1 @ m3

    def test_transpose(self):
        """Тестирование транспонирования матрицы"""
        m = array([[1, 2, 3], [4, 5, 6]])

        result = m.T
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data, [[1, 4], [2, 5], [3, 6]])

        result = m.T
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data, [[1, 4], [2, 5], [3, 6]])

    def test_det(self):
        """Тестирование вычисления определителя"""

        m1 = array([[5]])
        self.assertEqual(m1.det, 5)

        m2 = array([[1, 2], [3, 4]])
        self.assertEqual(m2.det, -2)

        m3 = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m3.det, 0)

        m4 = array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m4.det

    def test_inverse(self):
        """Тестирование вычисления обратной матрицы"""

        m = array([[4, 7], [2, 6]])
        inv = m.inv
        self.assertEqual(inv.shape, (2, 2))

        result = m @ inv
        for i in range(2):
            for j in range(2):
                if i == j:
                    self.assertAlmostEqual(result.data[i][j], 1, places=10)
                else:
                    self.assertAlmostEqual(result.data[i][j], 0, places=10)

        m_singular = array([[1, 2], [2, 4]])
        with self.assertRaises(ValueError):
            m_singular.inv

        m_non_square = array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m_non_square.inv

    def test_rank(self):
        """Тестирование вычисления ранга матрицы"""

        m1 = array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(m1.rank, 3)

        m2 = array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        self.assertEqual(m2.rank, 1)

        m3 = array([[0, 0], [0, 0]])
        self.assertEqual(m3.rank, 0)

    def test_trace(self):
        """Тестирование вычисления следа матрицы"""
        m = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m.trace, 15)

        m_non_square = array([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError):
            m_non_square.trace

    def test_diag(self):
        """Тестирование получения диагонали матрицы"""
        m = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m.diag, [1, 5, 9])

        m_non_square = array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m_non_square.diag, [1, 5])


if __name__ == "__main__":
    unittest.main()
