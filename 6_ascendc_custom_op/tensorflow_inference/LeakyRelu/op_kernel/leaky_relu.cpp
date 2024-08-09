#include "kernel_operator.h"
#include "kernel_leaky_relu.h"
#define UB_LIMIT ((AscendC::TOTAL_UB_SIZE) / 4 / sizeof(half))

extern "C" __global__ __aicore__ void leaky_relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // kernel impl
    AscendC::ElemwiseFrame<leaky_relu_ascendc::KernelLeakyRelu<half>> op;
    InitTilingParam<UB_LIMIT>(tiling_data.size, op.param);
    op.Init(x, y);
    op.Process();
}
