import cugrad

def main():
    print("Creating Value instances:")
    a = cugrad.Value(2.0)
    b = cugrad.Value(-3.0)
    c = cugrad.Value(10.0)
    
    print("a =", a)  # <Value data=2.0, grad=0.0>
    print("b =", b)  # <Value data=-3.0, grad=0.0>
    print("c =", c)  # <Value data=10.0, grad=0.0>
    
    print("\nPerforming arithmetic operations:")
    d = a * b + c
    print("d = a * b + c =", d)  # Expected: <Value data=4.0, grad=-3.0>
    
    # Accessing attributes
    print("\nAccessing attributes of d:")
    print("d.data =", d.data)  # 4.0
    print("d.grad =", d.grad)  # -3.0
    print("d._op =", d._op)    # '+'
    
    # Additional operations
    print("\nAdditional operations:")
    e = d / a
    print("e = d / a =", e)  # <Value data=2.0, grad=1.5>
    
if __name__ == "__main__":
    main()
