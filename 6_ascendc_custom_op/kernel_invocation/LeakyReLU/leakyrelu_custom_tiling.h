#ifndef LEAKYRELU_CUSTOM_TILING_H
#define LEAKYRELU_CUSTOM_TILING_H

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignDiv32(int32_t n)
{
    return ((n + 31) & ~31) / 32;
}

struct LeakyReluCustomTilingData
{
    uint32_t totalLength;
    uint32_t tileNum;
    float scalar;
};

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#ifdef __CCE_KT_TEST__
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);
#else

#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)                          \
    __ubuf__ uint8_t *tilingUbPointer = (__ubuf__ uint8_t *)get_imm(0);                           \
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)(tilingUbPointer)), ((__gm__ uint8_t*)(tilingPointer)), \
                    0, 1, AlignDiv32(sizeof(tilingStruct)), 0, 0);                                \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingUbPointer);                        \
    pipe_barrier(PIPE_ALL);
#endif

#define GET_TILING_DATA(tilingData, tilingPointer)                                 \
    LeakyReluCustomTilingData tilingData;                                          \
    INIT_TILING_DATA(LeakyReluCustomTilingData, tilingDataPointer, tilingPointer); \
    (tilingData).totalLength = tilingDataPointer->totalLength;                     \
    (tilingData).tileNum = tilingDataPointer->tileNum;                             \
    (tilingData).scalar = tilingDataPointer->scalar;
#endif
