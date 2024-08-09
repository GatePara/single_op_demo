/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef KERNEL_LEAKY_RELU_H
#define KERNEL_LEAKY_RELU_H
#include "op_frame/elemwise_frame.h"
#include "kernel_leaky_relu_tiling.h"

namespace leaky_relu_ascendc {
template <typename T> class KernelLeakyRelu : public AscendC::ElemwiseOpBase {
public:
    using DType = T;
    __aicore__ KernelLeakyRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);

public:
    __aicore__ inline void MyCopyIn(int32_t progress, AscendC::LocalTensor<T>& inBuf);
    __aicore__ inline void MyCompute(int32_t progress, AscendC::LocalTensor<T>& inBuf, AscendC::LocalTensor<T>& outBuf);
    __aicore__ inline void MyCopyOut(int32_t progress, AscendC::LocalTensor<T>& outBuf);

public:
    LeakyReluParam param;

private:
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
};

template <typename T> __aicore__ inline void KernelLeakyRelu<T>::Init(GM_ADDR x, GM_ADDR y)
{
    ElemwiseOpBase::Init(param.loopSize, param.dataLen, 0, param.dataLen);
    xGm.SetGlobalBuffer((__gm__ T*)(x) + block_idx * param.blockFactor);
    yGm.SetGlobalBuffer((__gm__ T*)(y) + block_idx * param.blockFactor);
}

template <typename T>
__aicore__ inline void KernelLeakyRelu<T>::MyCopyIn(int32_t progress, AscendC::LocalTensor<T>& x_buf)
{
    auto tailFlag = 0;
    if (param.loopSize == progress + 1) {
        tailFlag = 1;
    }
    x_buf.SetUserTag(tailFlag);
    AscendC::DataCopy(x_buf, xGm[progress * param.ubFactor], param.dmaParam[tailFlag]);
}

template <typename T>
__aicore__ inline void KernelLeakyRelu<T>::MyCompute(int32_t progress, AscendC::LocalTensor<T>& x_buf,
    AscendC::LocalTensor<T>& y_buf)
{
    auto x_tag = x_buf.GetUserTag();
    y_buf.SetUserTag(x_tag);
    AscendC::LeakyRelu(y_buf, x_buf, param.negativeSlope, param.itemSize[x_tag]);
}

template <typename T>
__aicore__ inline void KernelLeakyRelu<T>::MyCopyOut(int32_t progress, AscendC::LocalTensor<T>& y_buf)
{
    AscendC::DataCopy(yGm[progress * param.ubFactor], y_buf, param.dmaParam[y_buf.GetUserTag()]);
}
} // namespace leaky_relu
#endif // KERNEL_LEAKY_RELU_H