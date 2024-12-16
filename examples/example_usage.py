# example_usage.py

from cugrad.nn import Layer
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
    x = [Tensor(2.0), Tensor(3.0)]
    n = Layer(2,3)
    
    res = n(x)
    for t in res:
        print(t)

if __name__ == "__main__":
    main()
