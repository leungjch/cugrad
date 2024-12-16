# example_usage.py

from cugrad.nn import Linear
from cugrad.tensor import Tensor

def main():
    a = Tensor(2.0); a.label = 'a'
    b = Tensor(3.0); b.label = 'b'
    c = Tensor(1.0); c.label = 'c'
    d = a * b; d.label = 'd'
    e = d + c; e.label = 'e'
    e.backward()

    def dfs(t):
        print(t)
        for child in t.children:
            dfs(child)

    dfs(e)

    # Example usage of Linear
    linear = Linear(1, 1)
    input_tensor = Tensor(5.0)
    output = linear.forward([input_tensor])
    print(output)

if __name__ == "__main__":
    main()
