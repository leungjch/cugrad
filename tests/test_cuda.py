import unittest
import numpy as np
from cugrad.tensor import Tensor
from cugrad import set_device, DeviceType

class TestCUDA(unittest.TestCase):
    def test_cuda_addition(self):
        # Switch to CUDA device if available
        # (Make sure you compiled with CUDA support)
        set_device(DeviceType.CUDA)

        # Create two tensors on CPU
        size = 1000
        a = Tensor([size])
        b = Tensor([size])

        # Fill with random data
        # Using NumPy for convenience
        np_a = np.random.rand(size).astype(np.float32)
        np_b = np.random.rand(size).astype(np.float32)
        a.data = np_a.tolist()
        b.data = np_b.tolist()

        # Move them to CUDA
        a.to_device(DeviceType.CUDA)
        b.to_device(DeviceType.CUDA)

        # Perform an operation on CUDA
        c = a + b
        # Backward pass on CUDA
        c.backward()

        # Move results back to CPU to check
        a.to_device(DeviceType.CPU)
        b.to_device(DeviceType.CPU)
        c.to_device(DeviceType.CPU)

        # Check correctness
        np_c = np.array(c.data)
        # Forward check: c = a + b
        np.testing.assert_allclose(np_c, np_a + np_b, rtol=1e-5, atol=1e-5)

        # Backward check:
        # dc/da = 1 and dc/db = 1, so after backward
        # a.grad and b.grad should be all ones
        np_a_grad = np.array(a.grad)
        np_b_grad = np.array(b.grad)
        np.testing.assert_allclose(np_a_grad, np.ones_like(np_a_grad), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np_b_grad, np.ones_like(np_b_grad), rtol=1e-5, atol=1e-5)

        # Switch back to CPU device
        set_device(DeviceType.CPU)

if __name__ == '__main__':
    unittest.main()
