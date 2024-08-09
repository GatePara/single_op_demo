1、环境变量依赖
   需要配置ASCEND_HOME_DIR，值为CANN包安装路径，例如export ASCEND_HOME_DIR=~/Ascend/ascend-toolkit/latest

2、CPU调试
    测试命令
    ```
    chmod +x run.sh
    ./run.sh leakyrelu_custom ascend910 VectorCore cpu
    ```

3、NPU调试
    测试命令
    ```
    chmod +x run.sh
    ./run.sh leakyrelu_custom ascend910 VectorCore npu
    ```
