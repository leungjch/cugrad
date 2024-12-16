// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <memory>
#include "tensor.h"
#include "nn.h" // Include the nn header to bind Linear

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

     // Bind the Module base class (if you want to expose it)
     py::class_<Module, std::shared_ptr<Module>>(nn, "Module")
         .def("forward", &Module::forward, "Forward pass")
         .def("zero_grad", &Module::zero_grad, "Zero gradients")
         .def("parameters", &Module::parameters, "Get parameters");

     // Bind the Linear class to the 'nn' submodule
     py::class_<Linear, Module, std::shared_ptr<Linear>>(nn, "Linear")
         .def(py::init<int, int>(), py::arg("in_features"), py::arg("out_features"), "Linear layer constructor")
         .def_readwrite("weights", &Linear::weights, "Weights of the linear layer")
         .def_readwrite("bias", &Linear::bias, "Bias of the linear layer")
         .def_readwrite("activation", &Linear::activation, "Activation operation")
         .def_readwrite("in_features", &Linear::in_features, "Number of input features")
         .def_readwrite("out_features", &Linear::out_features, "Number of output features")
         .def("forward", &Linear::forward, py::arg("input"), "Forward pass of the linear layer")
         .def("zero_grad", &Module::zero_grad, "Zero gradients")
         .def("parameters", &Module::parameters, "Get parameters");
}
