/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#include <chrono>
#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void matmul_custom_do(uint32_t coreDim, void* l2ctrl, void* stream,
    uint8_t *param1, uint8_t *param2, uint8_t *param3, uint8_t *param4);
#else
#include "tikicpulib.h"
extern "C" void matmul_custom(uint8_t *param1, uint8_t *param2, uint8_t *param3, uint8_t *param4);
#endif

int32_t main(int32_t argc, char* argv[])
{
    size_t param1FileSize = 512 * 512 * sizeof(uint16_t);  // uint16_t represent half
    size_t param2FileSize = 512 * 1024 * sizeof(uint16_t);  // uint16_t represent half
    size_t param3FileSize = 512 * 1024 * sizeof(float);
    size_t param4FileSize = 28 * sizeof(uint32_t);
    uint32_t blockDim = 1;

#ifdef __CCE_KT_TEST__
    uint8_t *param1 = (uint8_t *)AscendC::GmAlloc(param1FileSize);
    uint8_t *param2 = (uint8_t *)AscendC::GmAlloc(param2FileSize);
    uint8_t *param3 = (uint8_t *)AscendC::GmAlloc(param3FileSize);
    uint8_t *param4 = (uint8_t *)AscendC::GmAlloc(param4FileSize);

    ReadFile("./input/x1_gm.bin", param1FileSize, param1, param1FileSize);
    // PrintData(param1, 16, printDataType::HALF);
    ReadFile("./input/x2_gm.bin", param2FileSize, param2, param2FileSize);
    // PrintData(param2, 16, printDataType::HALF);
    ReadFile("./input/tiling.bin", param4FileSize, param4, param4FileSize);
    // PrintData(param4, 16, printDataType::UINT32_T);

    ICPU_RUN_KF(matmul_custom, blockDim, param1, param2, param3, param4);

    // PrintData(param3, 16, printDataType::FLOAT);
    WriteFile("./output/output.bin", param3, param3FileSize);

    AscendC::GmFree((void *)param1);
    AscendC::GmFree((void *)param2);
    AscendC::GmFree((void *)param3);
    AscendC::GmFree((void *)param4);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *param1Host;
    uint8_t *param1Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param1Host), param1FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param1Device, param1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", param1FileSize, param1Host, param1FileSize);
    // PrintData(param1Host, 16, printDataType::HALF);
    CHECK_ACL(aclrtMemcpy(param1Device, param1FileSize, param1Host, param1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *param2Host;
    uint8_t *param2Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param2Host), param2FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param2Device, param2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", param2FileSize, param2Host, param2FileSize);
    // PrintData(param2Host, 16, printDataType::HALF);
    CHECK_ACL(aclrtMemcpy(param2Device, param2FileSize, param2Host, param2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *param4Host;
    uint8_t *param4Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param4Host), param4FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param4Device, param4FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/tiling.bin", param4FileSize, param4Host, param4FileSize);
    // PrintData(param4Host, 16, printDataType::UINT32_T);
    CHECK_ACL(aclrtMemcpy(param4Device, param4FileSize, param4Host, param4FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *param3Host;
    uint8_t *param3Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param3Host), param3FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param3Device, param3FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    matmul_custom_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, param4Device);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(param3Host, param3FileSize, param3Device, param3FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // PrintData(param3Host, 16, printDataType::FLOAT);
    WriteFile("./output/output.bin", param3Host, param3FileSize);
    CHECK_ACL(aclrtFree(param3Device));
    CHECK_ACL(aclrtFreeHost(param3Host));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}