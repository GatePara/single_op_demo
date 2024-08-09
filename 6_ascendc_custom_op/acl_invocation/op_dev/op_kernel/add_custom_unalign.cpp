/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : z = x + y
 * This sample is a very basic sample that implements vector add on Ascend plaform.
 */
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t formerNum,
        uint32_t tailNum, uint32_t formerLength, uint32_t tailLength, uint32_t alignNum)
    {
        if (GetBlockIdx() < formerNum) {
            this->tileLength = formerLength;
            xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + formerLength * GetBlockIdx(), formerLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + formerLength * GetBlockIdx(), formerLength);
            zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + formerLength * GetBlockIdx(), formerLength);
        } else {
            this->tileLength = tailLength;
            xGm.SetGlobalBuffer(
                (__gm__ DTYPE_X *)x + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            yGm.SetGlobalBuffer(
                (__gm__ DTYPE_Y *)y + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
            zGm.SetGlobalBuffer(
                (__gm__ DTYPE_Z *)z + formerLength * formerNum + tailLength * (GetBlockIdx() - formerNum), tailLength);
        }
        ASSERT(alignNum != 0 && "align num can not be zero!");
        pipe.InitBuffer(inQueueX, BUFFER_NUM,
            (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM,
            (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM,
            (((this->tileLength + alignNum - 1) / alignNum) * alignNum) * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        DataCopy(xLocal, xGm, this->tileLength);
        DataCopy(yLocal, yGm, this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        DataCopy(zGm, zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom_unalign(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAdd op;
    op.Init(x, y, z, tilingData.formerNum, tilingData.tailNum, tilingData.formerLength,
        tilingData.tailLength, tilingData.alignNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }
}
