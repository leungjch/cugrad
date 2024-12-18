import unittest
from cugrad.tensor import Tensor

class TestValue(unittest.TestCase):
    def test_initialization(self):
        v = Tensor()
        self.assertEqual(v.data, 0.0)
        self.assertEqual(v.grad, 0.0)

        v = Tensor(5.0)
        self.assertEqual(v.data, 5.0)
        self.assertEqual(v.grad, 0.0)
        self.assertIsNone(v.op)


    def test_addition(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a + b
        self.assertEqual(c.data, 5.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c.op.op_type, 'add')

    def test_subtraction(self):
        a = Tensor(5.0)
        b = Tensor(3.0)
        c = a - b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c.op.op_type, 'sub')

    def test_multiplication(self):
        a = Tensor(2.0)
        b = Tensor(3.0)
        c = a * b
        self.assertEqual(c.data, 6.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c.op.op_type, 'mul')

    def test_division(self):
        a = Tensor(6.0)
        b = Tensor(3.0)
        c = a / b
        self.assertEqual(c.data, 2.0)
        self.assertEqual(c.grad, 0.0)
        self.assertEqual(c.op.op_type, 'div')

    def test_reverse_operations(self):
        a = Tensor(3.0)
        b = 2.0 * a
        self.assertEqual(b.data, 6.0)
        
        c = 5.0 + a
        self.assertEqual(c.data, 8.0)
        
        d = 10.0 - a
        self.assertEqual(d.data, 7.0)
        
        e = 6.0 / a
        self.assertEqual(e.data, 2.0)

if __name__ == "__main__":
    unittest.main()
