import unittest
from cugrad.tensor import Tensor
from cugrad import DeviceType, set_device

set_device(DeviceType.CPU)

class TestTensor(unittest.TestCase):
    def test_addition(self):
        a = Tensor([[1.0,2.0],[3.0,4.0]])
        b = Tensor([[5.0,6.0],[7.0,8.0]])
        c = a + b
        c.to_device(DeviceType.CPU)
        self.assertEqual(c.shape, [2,2])
        expected = [6.0,8.0,10.0,12.0]
        self.assertEqual(c.data, expected)

    def test_subtraction(self):
        a = Tensor([5.0])
        b = Tensor([3.0])
        c = a - b
        c.to_device(DeviceType.CPU)
        self.assertEqual(c.data, [2.0])

    def test_multiplication(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([3.0, 4.0])
        c = a * b
        c.to_device(DeviceType.CPU)
        self.assertEqual(c.data, [6.0, 12.0])

    def test_division(self):
        a = Tensor([6.0])
        b = Tensor([3.0])
        c = a / b
        c.to_device(DeviceType.CPU)
        self.assertEqual(c.data, [2.0])

    def test_catch_shape_mismatch(self):
        a = Tensor([1])
        b = Tensor([2,1])
        with self.assertRaises(ValueError):
            c = a + b

if __name__ == "__main__":
    unittest.main()
