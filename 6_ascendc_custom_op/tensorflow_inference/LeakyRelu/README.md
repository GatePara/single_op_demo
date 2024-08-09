AscendC 自定义算子入TensorFlow网络示例教程:
以Yolov3 TensorFlow离线推理为例
推理平台：Ascend310P3

一、自定义算子准备
1.先构建AscendC-LeakyRelu算子工程
/usr/local/python3.7/bin/msopgen gen -i leakyrelu.json -f tf  -c ai_core-ascend310p -lan cpp -out ./custom_op
2.将目录下的op_host和op_kernel实现同步至生成的custom_op工程目录下,可以替换之前msopgen生成的默认文件
3.确认CMakePresets.json中 "ASCEND_CANN_PACKAGE_PATH" 为CANN软件包安装路径，执行 ./build.sh编译出自定义算子包
4.安装在custom_op/build_out/目录下生成的自定义算子run包

二、离线推理验证流程
1.先下载yolov3 tensorflow离线pb模型：
https://gitee.com/link?target=https%3A%2F%2Fobs-9be7.obs.cn-east-2.myhuaweicloud.com%2F003_Atc_Models%2Fmodelzoo%2Fyolov3_tf.pb

2.Pb模型转换为om模型
For Ascend310P3:
atc --model=./yolov3_tf.pb --framework=3 --output=./YOLOv3_TF --input_shape="input:4,416,416,3" --soc_version=Ascend310P3 --fusion_switch_file=fusion_off.cfg
其中 --fusion_switch_file为关闭算子融合配置，此处若不关闭融合，LeakyRelu算子会进行融合，因此会无法单独编译LeakyRelu算子进行验证

若出现:
start compile Ascend C operator LeakyRelu. kernel name is leaky_relu
compile Ascend C operator: LeakyRelu success!
打印，表明进入了AscendC算子编译

出现ATC run success, welcome to the next use 表明离线om模型转换成功

3.执行离线推理
可使用https://gitee.com/ascend/tools/tree/master/msame 该工具

