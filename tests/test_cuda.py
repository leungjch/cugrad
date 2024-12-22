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
        size = 3
        a = Tensor([size])
        b = Tensor([size])

        # Fill with [1,2,3] and [4,5,6]
        # Using NumPy for convenience
        np_a = np.array([1,2,3]).astype(np.float32)
        np_b = np.array([4,5,6]).astype(np.float32)
        a.data = np_a.tolist()
        b.data = np_b.tolist()

        # Move them to CUDA
        a.to_device(DeviceType.CUDA)
        b.to_device(DeviceType.CUDA)

        # Perform an operation on CUDA
        c = a + b   

        # c is created as a new tensor with device = cuda
        # # but the data is created on CPU
        # need to clean up this mismatch  

        # Backward pass on CUDA
        c.backward()

        # Move results back to CPU to check
        a.to_device(DeviceType.CPU)
        b.to_device(DeviceType.CPU)
        c.to_device(DeviceType.CPU)

        # Check correctness
        np_c = np.array(c.data)
        np_grad = np.array(c.grad)
        print("Data on CPU: ", np_c)
        print("Grad on CPU: ", np_grad)
        # Forward check: c = a + b
        np.testing.assert_allclose(np_c, np_a + np_b, rtol=1e-5, atol=1e-5)


        # Backward check:
        # dc/da = 1 and dc/db = 1, so after backward
        # a.grad and b.grad should be all ones
        np_a_grad = np.array(a.grad)
        np_b_grad = np.array(b.grad)
        print("np_a_grad: ", np_a_grad)
        print("np_b_grad: ", np_b_grad)
        np.testing.assert_allclose(np_a_grad, np.ones_like(np_a_grad), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np_b_grad, np.ones_like(np_b_grad), rtol=1e-5, atol=1e-5)

        # Switch back to CPU device
        set_device(DeviceType.CPU)

if __name__ == '__main__':
    unittest.main()
