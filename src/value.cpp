#include "value.h"

// Constructors
Value::Value(float data, float grad, char op) : data(data), grad(grad), _op(op) {}

Value::Value(float data) : data(data), grad(0.0f), _op(' ') {}

Value::Value() : data(0.0f), grad(0.0f), _op(' ') {}

// Operator Overloads
Value Value::operator+(const Value &other) const
{
    return Value(data + other.data, grad + other.grad, '+');
}

Value Value::operator-(const Value &other) const
{
    return Value(data - other.data, grad - other.grad, '-');
}

Value Value::operator*(const Value &other) const
{
    return Value(data * other.data, grad * other.data + data * other.grad, '*');
}

Value Value::operator/(const Value &other) const
{
    return Value(data / other.data,
                 grad / other.data - (data * other.grad) / (other.data * other.data),
                 '/');
}

// Friend function for ostream
std::ostream &operator<<(std::ostream &os, const Value &value)
{
    os << "Value(data=" << value.data << ", grad=" << value.grad << ")";
    return os;
}
