import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base import array


class TestMatrixBasic(unittest.TestCase):
    def test_init(self):
        """Тестирование инициализации матрицы"""
        data = [[1, 2, 3], [4, 5, 6]]
        m = array(data)
        self.assertEqual(m.shape, (2, 3))
        self.assertEqual(m.data, data)

        m = array(shape=(2, 3), fill_value=7)
        self.assertEqual(m.shape, (2, 3))
        for i in range(2):
            for j in range(3):
                self.assertEqual(m.data[i][j], 7)

        m1 = array(data)
        m2 = array(m1)
        self.assertEqual(m1.shape, m2.shape)
        self.assertEqual(m1.data, m2.data)
        m1.data[0][0] = 99
        self.assertNotEqual(m1.data[0][0], m2.data[0][0])

    def test_getitem_setitem(self):
        """Тестирование получения и установки элементов"""
        m = array([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[1, 2], 6)

        m[0, 0] = 10
        self.assertEqual(m[0, 0], 10)

        self.assertEqual(m[0], [10, 2, 3])

        sub_m = m[0:2, 1:3]
        self.assertEqual(sub_m.shape, (2, 2))
        self.assertEqual(sub_m.data, [[2, 3], [5, 6]])

    def test_eq(self):
        """Тестирование сравнения матриц"""
        m1 = array([[1, 2], [3, 4]])
        m2 = array([[1, 2], [3, 4]])
        m3 = array([[1, 2], [3, 5]])

        self.assertTrue(m1 == m2)
        self.assertFalse(m1 == m3)
        self.assertTrue(m1 == [[1, 2], [3, 4]])
        self.assertFalse(m1 == [[1, 2], [3, 5]])

    def test_copy(self):
        """Тестирование создания копии матрицы"""
        m1 = array([[1, 2], [3, 4]])
        m2 = m1.copy()

        self.assertEqual(m1.data, m2.data)
        m1[0, 0] = 99
        self.assertNotEqual(m1[0, 0], m2[0, 0])


if __name__ == "__main__":
    unittest.main()
