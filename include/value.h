#ifndef VALUE_H
#define VALUE_H

#include <iostream>

// Based on the Python code of "micrograd" by Andrej Karpathy
class Value
{
public:
    float data;
    float grad;
    char _op;

    // Constructors
    Value(float data, float grad, char op);
    Value(float data);
    Value();

    // Operator Overloads
    Value operator+(const Value &other) const;
    Value operator-(const Value &other) const;
    Value operator*(const Value &other) const;
    Value operator/(const Value &other) const;

    // Friend function for ostream
    friend std::ostream &operator<<(std::ostream &os, const Value &value);
};

#endif // VALUE_H
