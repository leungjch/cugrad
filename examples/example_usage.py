import cugrad

def main():
    a = cugrad.Tensor(2.0); a.label = 'a'
    b = cugrad.Tensor(3.0); b.label = 'b'
    c = cugrad.Tensor(1.0); c.label = 'c'
    d = a * b; d.label = 'd'
    e = d + c; e.label = 'e'
    e.backward()

    def dfs(t):
        print(t)
        for child in t.children:
            dfs(child)

    dfs(e)

    
if __name__ == "__main__":
    main()
