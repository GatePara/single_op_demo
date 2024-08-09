/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#include "add_custom_unalign_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t SIZE_OF_HALF = 2;
constexpr uint32_t BLOCK_SIZE = 32;
// shape需要对齐到的最小单位
constexpr uint32_t ALIGN_NUM = BLOCK_SIZE / SIZE_OF_HALF;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TilingDataUnalign tiling;
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    auto attrs = context->GetAttrs();
    const uint32_t* testAttrs = attrs->GetAttrPointer<uint32_t>(1);

    // 如果是非对齐的shape，需要向上对齐到最小单位
    uint32_t totalLengthAligned = ((totalLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    // 把所有的数据尽可能均匀地分配到每个核上，如果不能均分的话，那么会有部分核多算一个最小单位ALIGN_NUM
    // 通过模的计算，可以得到多算一个最小单位的核的数量，也可以得到少算一个最小单位的核的数量
    // eg：对齐后的总数据量为160，核心数为8，数据块的最小单位是16，那么：
    // 1、最小单位数据块的总数：160 / 16 = 10
    // 2、有2个核会分到2个最小单位的数据块：10 % 8 =2，可以称之为整块
    // 3、有6个核会分到1个最小单位的数据块：8 - 2 = 6，可以称之为尾块
    uint32_t formerNum = (totalLengthAligned / ALIGN_NUM) % BLOCK_DIM;
    uint32_t tailNum = BLOCK_DIM - formerNum;
    // 计算整块和尾块的数据量
    uint32_t formerLength = ((totalLengthAligned / BLOCK_DIM + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
    uint32_t tailLength = (totalLengthAligned / BLOCK_DIM / ALIGN_NUM) * ALIGN_NUM;

    tiling.set_formerNum(formerNum);
    tiling.set_tailNum(tailNum);
    tiling.set_formerLength(formerLength);
    tiling.set_tailLength(tailLength);
    tiling.set_alignNum(ALIGN_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetTilingKey(1);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto inputShape = context->GetInputShape(0);
    auto outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class AddCustomUnalign : public OpDef {
public:
    explicit AddCustomUnalign(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32 })
            .Format({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND });
        this->Attr("testAttr1")
            .AttrType(OPTIONAL)
            .Float(8.0);
        this->Attr("testAttr2")
            .AttrType(REQUIRED)
            .Int();
        this->SetInferShape(ge::InferShape);

        this->AICore().SetTiling(optiling::TilingFunc);

        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomUnalign);
} // namespace ops
