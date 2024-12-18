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
        "Add": "lightblue",
        "Subtract": "lightgreen",
        "Multiply": "orange",
        "Divide": "pink",
        "Exp": "yellow",
        "Tanh": "purple",
        # Add more operations and colors as needed
    }

    def add_nodes(tensor):
        if id(tensor) in tensor_id_to_node:
            return

        current_id = next(uid)
        tensor_node = f"tensor_{current_id}"
        tensor_id_to_node[id(tensor)] = tensor_node

        # Node label includes label, data, grad
        label = f"{tensor.label}\nData: {tensor.data:.2f}\nGrad: {tensor.grad:.2f}"
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
    dot.render(filename, view=False)
    print(f"Computation graph saved as {filename}.{format}")
