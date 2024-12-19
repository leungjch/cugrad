import unittest
from cugrad.tensor import Tensor

class TestValue(unittest.TestCase):
    def test_initialization(self):
        v = Tensor([2,2])
        self.assertEqual(v.shape, [2,2])
        self.assertEqual(len(v.data), 4)
        self.assertEqual(len(v.grad), 4)
        # Check default initialization
        for val in v.data:
            self.assertEqual(val, 0.0)
        for val in v.grad:
            self.assertEqual(val, 0.0)

    def test_addition(self):
        a = Tensor([2,2])
        a.data = [1.0,2.0,3.0,4.0]
        b = Tensor([2,2])
        b.data = [5.0,6.0,7.0,8.0]
        c = a + b
        self.assertEqual(c.shape, [2,2])
        expected = [6.0,8.0,10.0,12.0]
        self.assertEqual(c.data, expected)

    def test_subtraction(self):
        a = Tensor([1])
        a.data = [5.0]
        b = Tensor([1])
        b.data = [3.0]
        c = a - b
        self.assertEqual(c.data, [2.0])

    def test_multiplication(self):
        a = Tensor([2,1])
        a.data = [2.0, 3.0]
        b = Tensor([2,1])
        b.data = [3.0, 4.0]
        c = a * b
        self.assertEqual(c.data, [6.0, 12.0])

    def test_division(self):
        a = Tensor([1])
        a.data = [6.0]
        b = Tensor([1])
        b.data = [3.0]
        c = a / b
        self.assertEqual(c.data, [2.0])

    def test_catch_shape_mismatch(self):
        a = Tensor([2,2])
        b = Tensor([2,1])
        with self.assertRaises(ValueError):
            c = a + b

if __name__ == "__main__":
    unittest.main()
