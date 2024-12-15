import unittest
import cugrad

class TestValue(unittest.TestCase):
    def test_initialization(self):
        v = cugrad.Value()
        self.assertEqual(v.data, 0.0)
        self.assertEqual(v.grad, 0.0)
        self.assertEqual(v._op, ' ')

        v = cugrad.Value(5.0)
        self.assertEqual(v.data, 5.0)
        self.assertEqual(v.grad, 0.0)
        self.assertEqual(v._op, ' ')

        v = cugrad.Value(3.0, 1.5, '+')
        self.assertEqual(v.data, 3.0)
        self.assertEqual(v.grad, 1.5)
        self.assertEqual(v._op, '+')

    def test_addition(self):
        a = cugrad.Value(2.0)
        b = cugrad.Value(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '+')

    def test_subtraction(self):
        a = cugrad.Value(5.0)
        b = cugrad.Value(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '-')

    def test_multiplication(self):
        a = cugrad.Value(2.0)
        b = cugrad.Value(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '*')

    def test_division(self):
        a = cugrad.Value(6.0)
        b = cugrad.Value(3.0)
        c = a / b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c._op, '/')

    def test_repr(self):
        v = cugrad.Value(2.0, 1.0, '+')
        self.assertEqual(repr(v), "<Value data=2.000000, grad=1.000000>")

if __name__ == "__main__":
    unittest.main()
