### Demostration
```shell
bash run.sh [KERNEL_NAME](add_custom/matmul_custom/topk_custom) [SOC_VERSION](ascend910/ascend310p) [CORE_TYPE](AiCore/VectorCore) [RUN_MODE](cpu/npu)
```
### NOTICE
THEY ARE JUST DEMOS, NO DFX DEFENSE, do not type invalid command!!!
actually all that you can run:
#### On ascend910
```shell
(cd Add; bash run.sh add_custom ascend910 AiCore cpu)
(cd Add; bash run.sh add_custom ascend910 AiCore npu)

(cd Add_tile; bash run.sh add_custom ascend910 AiCore cpu)
(cd Add_tile; bash run.sh add_custom ascend910 AiCore npu)

(cd MatMul; bash run.sh matmul_custom ascend910 AiCore cpu)
(cd MatMul; bash run.sh matmul_custom ascend910 AiCore npu)
```

#### On ascend310p
```shell
(cd Add; bash run.sh add_custom ascend310p AiCore cpu)
(cd Add; bash run.sh add_custom ascend310p AiCore npu)

(cd MatMul; bash run.sh matmul_custom ascend310p AiCore cpu)
(cd MatMul; bash run.sh matmul_custom ascend310p AiCore npu)
```

