import unittest
from cugrad.tensor import Tensor
from cugrad import set_device, DeviceType, StackOp
import math
import numpy as np


set_device(DeviceType.CPU)

class TestGradients(unittest.TestCase):
    def test_gradient_addition(self):
        a = Tensor([2.0])
        b = Tensor([3.0])
        print("a: ", a.data)
        print("b: ", b.data)
        c = a + b
        print("c.operation: ", c.op)
        print("c: ", c.data)

        c.backward()

        c.to_device(DeviceType.CPU)
        a.to_device(DeviceType.CPU)
        b.to_device(DeviceType.CPU)

        self.assertEqual(c.data, [5.0])
        self.assertEqual(c.grad, [1.0])
        self.assertEqual(a.grad, [1.0])
        self.assertEqual(b.grad, [1.0])

    def test_gradient_multiplication(self):
        a = Tensor([2.0])
        b = Tensor([3.0])
        c = a * b
        c.backward()
        c.to_device(DeviceType.CPU)
        a.to_device(DeviceType.CPU)
        b.to_device(DeviceType.CPU)
        self.assertEqual(a.grad, [3.0])
        self.assertEqual(b.grad, [2.0])

    def test_gradient_tanh(self):
        a = Tensor([1]); a.label = "a"
        a.data = [0.5]
        b = Tensor([1]); b.label = "b"
        b.data = [0.25]
        sm = (a+b); sm.label = "sm"
        # sm.copy_to_host()
        c = sm.tanh(); c.label = "c"
        # c.backward()
        c.to_device(DeviceType.CPU)

        # derivative of tanh at 100 is 0
        self.assertAlmostEqual(c.grad[0], 0.0, places=5)

    def test_gradient_mlp(self):
        from cugrad.nn import MLP
        model = MLP(input_size=2, layer_sizes=[2, 1])

        # Create dummy input and target
        inputs = Tensor([1.0, 2.0])
        target = Tensor([1.0])

        print("inputs: ", inputs.device)
        print("target: ", target.device)

        output = model(inputs)  # output should be shape [1]
        print("output: ", output.data)
        print("target: ", target.data)
        print("output.device: ", output.device)
        print("output.grad before backward: ", output.grad)
        loss = (output - target)*(output - target)

        print("loss.grad before backward: ", loss.grad)
        print("loss.data before backward: ", loss.data)

        for param in model.parameters():
            # just ensure not all zero
            param.to_device(DeviceType.CPU)
            print("before param.data: ", param.data)
            print("before param.grad: ", param.grad)
            # param.to_device(DeviceType.CUDA) # move back to CUDA if CUDA is used


        loss.backward()
        
        loss.to_device(DeviceType.CPU)
        print("loss.grad after backward: ", loss.grad)
        print("loss.data after backward: ", loss.data)

        # Check that some parameter grads are non-zero
        for param in model.parameters():
            # just ensure not all zero
            param.to_device(DeviceType.CPU)
            print("param.data: ", param.data)
            print("param.grad: ", param.grad)
            self.assertTrue(any(g != 0.0 for g in param.grad))


    # -------------------- StackOp Tests --------------------
    
    def test_stackop_forward(self):
        """
        Test the forward pass of StackOp by stacking multiple [1]-shaped tensors
        and verifying the resulting [N]-shaped tensor.
        """
        # Create multiple [1]-shaped tensors
        t1 = Tensor([1.0])
        t2 = Tensor([2.0])
        t3 = Tensor([3.0])

        print("t1: ", t1.data)
        print("t2: ", t2.data)
        print("t3: ", t3.data)

        # Perform stacking
        stack_op = StackOp([t1, t2, t3])
        stack_op.forward()
        output = stack_op.output  # The stack operation returns a new Tensor

        # Move output to CPU to verify
        output.to_device(DeviceType.CPU)

        print("StackOp Forward Output: ", output.data)

        # Expected output: [1.0, 2.0, 3.0]
        expected = [1.0, 2.0, 3.0]
        self.assertEqual(output.data, expected)


    def test_stackop_backward(self):
        """
        Test the backward pass of StackOp by ensuring gradients are correctly
        propagated back to each input tensor.
        """
        # Set device to CUDA if available
        try:
            set_device(DeviceType.CUDA)
        except Exception as e:
            self.skipTest("CUDA device not available")
        
        # Create multiple [1]-shaped tensors
        t1 = Tensor([1])
        t1.data = [1.0]
        t2 = Tensor([1])
        t2.data = [2.0]
        t3 = Tensor([1])
        t3.data = [3.0]

        # Perform stacking including t1
        stack_op = StackOp([t1, t2, t3])
        stack_op.forward()
        output = stack_op.output

        # Perform backward pass
        output.backward()

        # Move tensors back to CPU to verify gradients
        t1.to_device(DeviceType.CPU)
        t2.to_device(DeviceType.CPU)
        t3.to_device(DeviceType.CPU)

        print("t1.data: ", t1.data)
        print("t2.data: ", t2.data)
        print("t3.data: ", t3.data)

        print("t1.grad after StackOp backward: ", t1.grad)
        print("t2.grad after StackOp backward: ", t2.grad)
        print("t3.grad after StackOp backward: ", t3.grad)

        # Each input should receive the corresponding gradient
        expected_grad = [1.0]
        self.assertEqual(t1.grad, expected_grad, "t1.grad should be [1.0]")
        self.assertEqual(t2.grad, expected_grad, "t2.grad should be [1.0]")
        self.assertEqual(t3.grad, expected_grad, "t3.grad should be [1.0]")

    # -------------------- SumOp Tests --------------------
    
    def test_sumop_forward(self):
        """
        Test the forward pass of SumOp by summing all elements of a tensor
        and verifying the scalar output.
        """
        # Create a [3]-shaped tensor
        t = Tensor([1.0, 2.0, 3.0])

        print("SumOp Input: ", t.data)

        # Perform summation
        sum_op = t.sum()  # Assuming a 'sum' method exists
        output = sum_op

        # Move output to CPU to verify
        output.to_device(DeviceType.CPU)

        print("SumOp Forward Output: ", output.data)

        # Expected output: [6.0]
        expected = [6.0]
        self.assertEqual(output.data, expected)

    def test_sumop_backward(self):
        """
        Test the backward pass of SumOp by ensuring gradients are correctly
        propagated back to the input tensor.
        """
        # Create a [3]-shaped tensor
        t = Tensor([1.0, 2.0, 3.0])

        # Perform summation
        sum_op = t.sum()  # Assuming a 'sum' method exists
        output = sum_op

        # Initialize output gradient
        output.grad = [1.0]

        # Perform backward pass
        output.backward()

        # Move tensor back to CPU to verify gradients
        t.to_device(DeviceType.CPU)

        print("t.grad after SumOp backward: ", t.grad)

        # Each element in the input should receive the gradient 1.0
        expected_grad = [1.0, 1.0, 1.0]
        self.assertEqual(t.grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
