// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <memory>
#include "tensor.h"

namespace py = pybind11;

// Helper function to enable operator overloading with shared_ptr<Tensor>
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
     py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
         // Constructors
         .def(py::init<>())                       // Default constructor
         .def(py::init<float>(), py::arg("data")) // Constructor with data

         // Properties
         .def_readwrite("data", &Tensor::data)
         .def_readwrite("grad", &Tensor::grad)
         .def_readwrite("children", &Tensor::children)
         .def_readwrite("label", &Tensor::label)

         // Methods
         .def("backward", &Tensor::backward, "Compute the gradients")
         .def("zero_grad", &Tensor::zero_grad, "Reset gradients to zero")

         // Operator Overloads
         .def("__add__", &tensor_add, py::is_operator())
         .def("__radd__", &tensor_add, py::is_operator())
         .def("__sub__", &tensor_sub, py::is_operator())
         .def("__rsub__", [](const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) -> std::shared_ptr<Tensor>
              {
                // Implement b - a
                return b->operator-(a); }, py::is_operator())
         .def("__mul__", &tensor_mul, py::is_operator())
         .def("__rmul__", &tensor_mul, py::is_operator())
         .def("__truediv__", &tensor_div, py::is_operator())
         .def("__rtruediv__", [](const std::shared_ptr<Tensor> &a, const std::shared_ptr<Tensor> &b) -> std::shared_ptr<Tensor>
              {
                // Implement b / a
                return b->operator/(a); }, py::is_operator())
         // Representation
         .def("__repr__", [](const Tensor &v)
              { return "<Tensor data=" + std::to_string(v.data) +
                       ", grad=" + std::to_string(v.grad) +
                       ", label=" + v.label + ">"; });
     ;
}
