// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include "value.h"

namespace py = pybind11;

PYBIND11_MODULE(cugrad, m)
{
     py::class_<Value>(m, "Value")
         .def(py::init<>())      // Default constructor
         .def(py::init<float>()) // Constructor with data
         .def(py::init<float, float, char>(),
              py::arg("data"), py::arg("grad"), py::arg("op")) // Full constructor
         .def_readwrite("data", &Value::data)
         .def_readwrite("grad", &Value::grad)
         .def_readwrite("_op", &Value::_op)
         // Operator Overloads
         .def(py::self + py::self)
         .def(py::self - py::self)
         .def(py::self * py::self)
         .def(py::self / py::self)
         // Representation
         .def("__repr__", [](const Value &v)
              { return "<Value data=" + std::to_string(v.data) +
                       ", grad=" + std::to_string(v.grad) + ">"; });
}
