/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */

#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR tilingGm)
{
    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }
    using A_T = half;
    using B_T = half;
    using C_T = float;
    using BiasT = float;

    TPipe que;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }

    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T*>(a), tiling.M * tiling.K);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T*>(b), tiling.K * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T*>(c), tiling.M * tiling.N);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];

    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T> aType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T> bType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> cType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasT> biasType;
    MatmulImpl<aType, bType, cType, biasType> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&tiling, &que);

    mm.SetTensorA(gmA);
    mm.SetTensorB(gmB);
    mm.IterateAll(gmC);
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint8_t* tilingGm)
{
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, tilingGm);
}
#endif
