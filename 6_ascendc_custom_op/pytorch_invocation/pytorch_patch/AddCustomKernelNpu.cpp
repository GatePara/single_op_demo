#include <torch/csrc/autograd/custom_function.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/aten/ops/op_api/op_api_common.h"

namespace at_npu {
namespace native {
using torch::autograd::Function;
using torch::autograd::AutogradContext;

at::Tensor NPUNativeFunctions::npu_add_custom(const at::Tensor& x, const at::Tensor& y) {
    at::Tensor result = OpPreparation::ApplyTensor(x); // 创建输出内存

    // calculate the output result of the NPU
    EXEC_NPU_CMD(aclnnAddCustom, x, y, result);
    return result;
}
} // namespace native
} // namespace at_npu
