// bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <memory>
#include "tensor.h"
#include "nn.h"
#include "op.h"
#include "optimizer.h"
#include "device_manager.h"

namespace py = pybind11;

PYBIND11_MODULE(cugrad, m)
{
    m.doc() = "cugrad: A CUDA-based automatic differentiation library";

    // Add a function to set the device from Python
    m.def("set_device", [](DeviceType device)
          { DeviceManager::get_instance().set_current_device(device); }, py::arg("device"), "Set the current device (CPU or CUDA)");

    m.def("get_device", []()
          { return DeviceManager::get_instance().get_current_device(); }, "Get the current device");

    // Bind the DeviceType enum
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("CUDA", DeviceType::CUDA)
        .export_values();

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

    py::class_<ReluOp, Op, std::shared_ptr<ReluOp>>(m, "ReluOp")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

    py::class_<SumOp, Op, std::shared_ptr<SumOp>>(m, "SumOp")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"));

    py::class_<StackOp, Op, std::shared_ptr<StackOp>>(m, "StackOp")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>> &>(), py::arg("inputs"))
        // forward()
        .def("forward", &StackOp::forward, "Forward pass")
        .def_readonly("output", &StackOp::output, "Output tensor");

    py::module tensor = m.def_submodule("tensor", "Tensor operations and classes");

    py::class_<Tensor, std::shared_ptr<Tensor>>(tensor, "Tensor")
        // Constructors
        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<int> &,
                      float,
                      std::shared_ptr<Op>,
                      std::vector<std::shared_ptr<Tensor>>>(),
             py::arg("shape"),
             py::arg("init_val") = 0.0f,
             py::arg("op") = nullptr, // or py::arg("op") = py::none()
             py::arg("children") = std::vector<std::shared_ptr<Tensor>>())
        // Custom constructor that accepts a NumPy array
        .def(py::init([](py::array_t<float> arr)
                      {
        // Request buffer info from NumPy array
        py::buffer_info buf = arr.request();
        int ndim = buf.ndim;

        // Construct shape vector
        std::vector<int> shape(ndim);
        for (int i = 0; i < ndim; i++)
        {
            shape[i] = static_cast<int>(buf.shape[i]);
        }

        // Compute size
        int size = 1;
        for (auto s : shape)
        {
            size *= s;
        }

        // Create a Tensor with the given shape (initialized to 0.0f)
        auto t = std::make_shared<Tensor>(shape,
                                          0.0f,
                                          nullptr,
                                          std::vector<std::shared_ptr<Tensor>>());

        // Copy data from the NumPy array into the tensor's data vector
        float *ptr = static_cast<float *>(buf.ptr);
        for (int i = 0; i < size; i++)
        {
            t->data[i] = ptr[i];
        }

        // Copy to device if necessary
        if (DeviceManager::get_instance().get_current_device() == DeviceType::CUDA)
        {
            t->copy_to_device();
        }

        return t; }),
             "Construct a Tensor from a NumPy array")

        // Properties
        .def_readwrite("data", &Tensor::data, "Tensor data")
        .def_readwrite("grad", &Tensor::grad, "Gradient of the tensor")
        .def_readwrite("shape", &Tensor::shape, "Shape of the tensor")
        .def_readwrite("children", &Tensor::children, "Child tensors")
        .def_readwrite("label", &Tensor::label, "Label for debugging")
        .def_readwrite("device", &Tensor::device, "Device type")
        .def_readonly("op", &Tensor::op, "Operation that created this tensor")

        // Methods
        .def("backward", &Tensor::backward, "Compute the gradients")
        .def("zero_grad", &Tensor::zero_grad, "Reset gradients to zero")

        // Device methods
        .def("allocate_memory_on_device", &Tensor::allocate_memory_on_device, "Allocate memory on the device")
        .def("to_device", &Tensor::to_device, "Move tensor to the specified device")
        .def("copy_to_device", &Tensor::copy_to_device, "Copy tensor to the device")
        .def("copy_to_host", &Tensor::copy_to_host, "Copy tensor to the host")

        // Operator Overloads
        .def("__add__", &operator+, py::is_operator())
        .def("__sub__", &operator-, py::is_operator())
        .def("__mul__", &operator*, py::is_operator())
        .def("__truediv__", &operator/, py::is_operator())

        // Right hand side operator overloads
        .def("__radd__", &operator+, py::is_operator())
        .def("__rsub__", &operator-, py::is_operator())
        .def("__rmul__", &operator*, py::is_operator())
        .def("__rtruediv__", &operator/, py::is_operator())

        // Other operations
        .def("tanh", &Tensor::tanh, "Apply the tanh operation")
        .def("relu", &Tensor::relu, "Apply the ReLU operation")
        .def("exp", &Tensor::exp, "Apply the exponential operation")
        .def("sum", &Tensor::sum, "Sum all elements of the tensor");

    py::module optimizer = m.def_submodule("optimizer", "Optimization algorithms");

    // Bind the SGD class
    py::class_<SGD, std::shared_ptr<SGD>>(optimizer, "SGD")
        .def(py::init<const std::vector<std::shared_ptr<Tensor>> &, float>(), py::arg("parameters"), py::arg("lr"), "SGD constructor with parameters and learning rate")
        .def("step", &SGD::step, "Update parameters")
        .def("zero_grad", &SGD::zero_grad, "Zero gradients");

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
        .def(py::init<int, int, bool>(), py::arg("input_size"), py::arg("output_size"), py::arg("nonlin") = true, "Layer constructor")
        .def("__call__", &Layer::operator(), py::arg("input"), "Call operator for the Layer");

    // Bind the MLP class to the 'nn' submodule
    py::class_<MLP, Module, std::shared_ptr<MLP>>(nn, "MLP")
        .def(py::init<int, const std::vector<int> &>(), py::arg("input_size"), py::arg("layer_sizes"), "MLP constructor with input size and layer sizes")
        .def("__call__", &MLP::operator(), py::arg("input"), "Call operator for the MLP")
        .def("parameters", &MLP::parameters, "Get all parameters of the MLP");
}
