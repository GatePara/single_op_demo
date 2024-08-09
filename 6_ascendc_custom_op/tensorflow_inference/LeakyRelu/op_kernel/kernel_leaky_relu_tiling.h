/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef KERNEL_LEAKY_RELU_TILING_H
#define KERNEL_LEAKY_RELU_TILING_H
#include "kernel_operator.h"

struct LeakyReluParam {
    uint32_t blockFactor;
    uint32_t ubFactor;
    int32_t dataLen;
    half negativeSlope;
    AscendC::DataCopyParams dmaParam[2] { {}, {} };
    uint32_t itemSize[2];
    int32_t loopSize;
};

template <int32_t limit>
__aicore__ inline void InitTilingParam(int32_t totalSize, LeakyReluParam& param, half slope = static_cast<half>(0.1))
{
    int32_t splitSize = totalSize / block_num;
    int64_t blockFactor = splitSize;

    const auto vec_len = AscendC::DEFAULT_BLOCK_SIZE / sizeof(half);

    int64_t ubFactor = blockFactor;
    int64_t blockNum = (splitSize + blockFactor - 1) / blockFactor;

    int64_t ub_for_num = (ubFactor + limit - 1) / limit;
    int64_t adjust_factor = (ubFactor + ub_for_num - 1) / ub_for_num;
    int64_t align_factor = (adjust_factor + vec_len - 1) / vec_len;

    ubFactor = align_factor * vec_len;
    if (ubFactor > limit) {
        ubFactor = (adjust_factor / vec_len) * vec_len;
    }
    param.negativeSlope = slope;
    param.blockFactor = blockFactor;
    param.ubFactor = ubFactor;
    param.loopSize = (blockFactor + ubFactor - 1) / ubFactor;
    param.dataLen = limit * sizeof(half);

    param.itemSize[0] = ubFactor;
    param.itemSize[1] = splitSize % ubFactor;
    param.dmaParam[0].blockLen = (ubFactor * sizeof(half) + AscendC::DEFAULT_C0_SIZE - 1) / AscendC::DEFAULT_C0_SIZE;
    param.dmaParam[1].blockLen =
        (param.itemSize[1] * sizeof(half) + AscendC::DEFAULT_C0_SIZE - 1) / AscendC::DEFAULT_C0_SIZE;

    if (param.itemSize[1] == 0) {
        param.itemSize[1] = ubFactor;
        param.dmaParam[1].blockLen = param.dmaParam[0].blockLen;
    }
};
#endif // KERNEL_LEAKY_RELU_TILING_H
