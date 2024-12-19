import graphviz
import itertools

def draw_compute_graph(tensor, filename="compute_graph", format="png"):

    dot = graphviz.Digraph(format=format)
    dot.attr(rankdir='LR', size='12,12')

    # Unique identifier generator
    uid = itertools.count()

    # Mapping from tensor id to node name to avoid duplicates
    tensor_id_to_node = {}
    op_id_to_node = {}

    # Define colors for different operations
    op_colors = {
        "add": "lightblue",
        "sub": "lightgreen",
        "mul": "orange",
        "div": "pink",
        "exp": "yellow",
        "tanh": "salmon",
        # Add more operations and colors as needed
    }

    def add_nodes(tensor):
        if id(tensor) in tensor_id_to_node:
            return

        current_id = next(uid)
        tensor_node = f"tensor_{current_id}"
        tensor_id_to_node[id(tensor)] = tensor_node

        def format(li):
            return '[' +  ','.join(f"{x:.2f}" for x in li) + ']'

        # Node label includes label, data, grad
        label = f"{tensor.label}\nData: {format(tensor.data)}\nGrad: {format(tensor.grad)}"
        dot.node(tensor_node, label=label, shape='ellipse')

        if tensor.op:
            # Create an operation node
            op = tensor.op
            op_node = f"op_{next(uid)}"
            op_id_to_node[id(op)] = op_node
            op_label = op.op_type
            color = op_colors.get(op_label, "gray")  # Default color if op_type not found
            dot.node(op_node, label=op_label, shape='box', style='filled', fillcolor=color)

            # Connect operation to tensor
            dot.edge(op_node, tensor_node)

            # Connect child tensors to operation
            for child in tensor.children:
                add_nodes(child)
                child_node = tensor_id_to_node[id(child)]
                dot.edge(child_node, op_node)

    add_nodes(tensor)
    # Make it large and high dpi
    dot.attr(size='20,20')
    dot.attr(dpi='800')
    dot.render(filename=filename, format=format, cleanup=True)
    print(f"Computation graph saved as {filename}.{format}")
