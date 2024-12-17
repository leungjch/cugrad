// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <memory>
#include "tensor.h"
#include "nn.h"
#include "op.h" // Include Op classes

namespace py = pybind11;

// Helper functions for operator overloading
std::shared_ptr<Tensor> tensor_add(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
     return a->operator+(b);
}

std::shared_ptr<Tensor> tensor_sub(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
     return a->operator-(b);
}

std::shared_ptr<Tensor> tensor_mul(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
     return a->operator*(b);
}

std::shared_ptr<Tensor> tensor_div(const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b)
{
     return a->operator/(b);
}

PYBIND11_MODULE(cugrad, m)
{
     m.doc() = "cugrad: A CUDA-based automatic differentiation library";

     // Create the 'tensor' submodule
     py::module tensor = m.def_submodule("tensor", "Tensor operations and classes");

     // Bind the Op base class
     py::class_<Op, std::shared_ptr<Op>>(m, "Op")
         .def_readonly("op_type", &Op::op_type, "Type of operation");

     // Bind derived Op classes
     py::class_<AddOp, Op, std::shared_ptr<AddOp>>(m, "AddOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     py::class_<SubtractOp, Op, std::shared_ptr<SubtractOp>>(m, "SubtractOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     py::class_<MultiplyOp, Op, std::shared_ptr<MultiplyOp>>(m, "MultiplyOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     py::class_<DivideOp, Op, std::shared_ptr<DivideOp>>(m, "DivideOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     py::class_<ExpOp, Op, std::shared_ptr<ExpOp>>(m, "ExpOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     py::class_<TanhOp, Op, std::shared_ptr<TanhOp>>(m, "TanhOp")
         .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

     // Bind the Tensor class to the 'tensor' submodule
     py::class_<Tensor, std::shared_ptr<Tensor>>(tensor, "Tensor")
         // Constructors
         .def(py::init<>(), "Default constructor")
         .def(py::init<float>(), py::arg("data"), "Constructor with data")

         // Properties
         .def_readwrite("data", &Tensor::data, "Tensor data")
         .def_readwrite("grad", &Tensor::grad, "Gradient of the tensor")
         .def_readwrite("children", &Tensor::children, "Child tensors")
         .def_readwrite("label", &Tensor::label, "Label for debugging")
         .def_readonly("op", &Tensor::op, "Operation that created this tensor")

         // Methods
         .def("backward", &Tensor::backward, "Compute the gradients")
         .def("zero_grad", &Tensor::zero_grad, "Reset gradients to zero")

         // Operator Overloads
         .def("__add__", &tensor_add, py::is_operator())
         .def("__radd__", &tensor_add, py::is_operator())
         .def("__sub__", &tensor_sub, py::is_operator())
         .def("__rsub__", [](const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) -> std::shared_ptr<Tensor>
              { return b->operator-(a); }, py::is_operator())
         .def("__mul__", &tensor_mul, py::is_operator())
         .def("__rmul__", &tensor_mul, py::is_operator())
         .def("__truediv__", &tensor_div, py::is_operator())
         .def("__rtruediv__", [](const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) -> std::shared_ptr<Tensor>
              { return b->operator/(a); }, py::is_operator())

         // Representation
         .def("__repr__", [](const Tensor &v)
              { return "<Tensor data=" + std::to_string(v.data) +
                       ", grad=" + std::to_string(v.grad) +
                       ", label=" + v.label + ">"; });

     // Create the 'nn' submodule
     py::module nn = m.def_submodule("nn", "Neural network modules");

     // Bind the Module base class
     py::class_<Module, std::shared_ptr<Module>>(nn, "Module")
         .def("__call__", &Module::operator(), "Call operator for the Module")
         .def("zero_grad", &Module::zero_grad, "Zero gradients")
         .def("parameters", &Module::parameters, "Get parameters");

     // Bind the Neuron class to the 'nn' submodule
     py::class_<Neuron, Module, std::shared_ptr<Neuron>>(nn, "Neuron")
         .def(py::init<int, bool>(), py::arg("in_features"), py::arg("nonlin"), "Neuron layer constructor")
         .def_readwrite("weights", &Neuron::weights, "Weights of the Neuron layer")
         .def_readwrite("bias", &Neuron::bias, "Bias of the Neuron layer")
         .def_readwrite("activation", &Neuron::activation, "Activation operation")
         .def_readwrite("in_features", &Neuron::in_features, "Number of input features")
         .def("zero_grad", &Module::zero_grad, "Zero gradients")
         .def("parameters", &Module::parameters, "Get parameters")
         .def("__call__", &Neuron::operator(), py::arg("input"), "Call operator for the Neuron layer");

     // Bind the Layer class to the 'nn' submodule
     py::class_<Layer, Module, std::shared_ptr<Layer>>(nn, "Layer")
         .def(py::init<int, int>(), py::arg("input_size"), py::arg("output_size"), "Layer constructor")
         .def("__call__", &Layer::operator(), py::arg("input"), "Call operator for the Layer");
}
