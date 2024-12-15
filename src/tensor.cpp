#include "tensor.h"
#include "op.h"

#include <memory>

#include "tensor.h"
#include "op.h"

std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor> &other)
{
    // Create a new AddOp with the current tensor and the other tensor as inputs
    auto add_op = std::make_shared<AddOp>(std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});

    // Perform the forward operation
    VALUE_TYPE result = add_op->forward();

    // Create the output tensor
    auto result_tensor = make_tensor(result, add_op, std::vector<std::shared_ptr<Tensor>>{shared_from_this(), other});
    add_op->output = result_tensor;

    return result_tensor;
}

// std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor> &other) const
// {
//     auto sub_op = std::make_shared<SubOp>();
//     return create_op(sub_op, shared_from_this(), other, &SubOp::forward);
// }

// std::shared_ptr<Tensor> Tensor::operator*(const std::shared_ptr<Tensor> &other) const
// {
//     auto mul_op = std::make_shared<MulOp>();
//     return create_op(mul_op, shared_from_this(), other, &MulOp::forward);
// }

// std::shared_ptr<Tensor> Tensor::operator/(const std::shared_ptr<Tensor> &other) const
// {
//     auto div_op = std::make_shared<DivOp>();
//     return create_op(div_op, shared_from_this(), other, &DivOp::forward);
// }
