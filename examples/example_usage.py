import cugrad

def main():
    print("Creating Tensor instances:")
    a = cugrad.Tensor(2.0)
    b = cugrad.Tensor(-3.0)
    c = cugrad.Tensor(10.0)
    
    print("a =", a)  # <Tensor data=2.0, grad=0.0>
    print("b =", b)  # <Tensor data=-3.0, grad=0.0>
    print("c =", c)  # <Tensor data=10.0, grad=0.0>
    
    print("\nPerforming arithmetic operations:")
    d = a * b + c
    print("d = a * b + c =", d)  # Expected: <Tensor data=4.0, grad=-3.0>
    
    # Accessing attributes
    print("\nAccessing attributes of d:")
    print("d.data =", d.data)  # 4.0
    print("d.grad =", d.grad)  # -3.0
    print("d._op =", d._op)    # '+'
    
    # Additional operations
    print("\nAdditional operations:")
    e = d / a
    print("e = d / a =", e)  # <Tensor data=2.0, grad=1.5>
    
if __name__ == "__main__":
    main()
