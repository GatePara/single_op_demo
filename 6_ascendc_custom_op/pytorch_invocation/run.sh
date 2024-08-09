#!/bin/bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

clear;clear
# 清除之前遗留的文件
rm -rf kernel_meta_temp* cache prof_total
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

# 导出环境变量
IS_DYNAMIC=$1
REPLAY_MODE=$2
PYTORCH_VERSION=1.11.0
PTA_DIR=pytorch-v${PYTORCH_VERSION}

if [ ! $ASCEND_HOME_DIR ]; then
    export ASCEND_HOME_DIR=/usr/local/Ascend/latest
fi
source $ASCEND_HOME_DIR/bin/setenv.bash

PYTHON_VERSION=`python3 -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1"."$2}'`
export HI_PYTHON=python${PYTHON_VERSION}
export PYTHONPATH=$ASCEND_HOME_DIR/python/site-packages:$PYTHONPATH
export PATH=$ASCEND_HOME_DIR/python/site-packages/bin:$PATH

# 检查当前昇腾芯片的类型
function check_soc_version() {
    SOC_VERSION_CONCAT=`python3 -c '''
import ctypes, os
def get_soc_version():
    max_len = 256
    rtsdll = ctypes.CDLL(f"libruntime.so")
    c_char_t = ctypes.create_string_buffer(b"\xff" * max_len, max_len)
    rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
    rt_error = rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
    if rt_error:
        print("rt_error:", rt_error)
        return ""
    soc_full_name = c_char_t.value.decode("utf-8")
    find_str = "Short_SoC_version="
    ascend_home_dir = os.environ.get("ASCEND_HOME_DIR")
    with open(f"{ascend_home_dir}/compiler/data/platform_config/{soc_full_name}.ini", "r") as f:
        for line in f:
            if find_str in line:
                start_index = line.find(find_str)
                result = line[start_index + len(find_str):].strip()
                return "{},{}".format(soc_full_name, result.lower())
    return ""
print(get_soc_version())
    '''`
    if [[ ${SOC_VERSION_CONCAT}"x" = "x" ]]; then
        echo "ERROR: SOC_VERSION_CONCAT is invalid!"
        return 1
    fi
    SOC_FULL_VERSION=`echo $SOC_VERSION_CONCAT | cut -d ',' -f 1`
    SOC_SHORT_VERSION=`echo $SOC_VERSION_CONCAT | cut -d ',' -f 2`
}

function main() {
    if [[ ${IS_DYNAMIC}"x" = "x" ]]; then
        echo "ERROR: IS_DYNAMIC is invalid!"
        return 1
    fi

    if [[ ${REPLAY_MODE}"x" = "x" || ${REPLAY_MODE} = "batch" || ${REPLAY_MODE} = "iterator" ]]; then
        echo "INFO: REPLAY_MODE valid : ${REPLAY_MODE}"
    else
        echo "ERROR: REPLAY_MODE is invalid!"
        return 1
    fi

    # 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf $ASCEND_OPP_PATH/vendors/*
    rm -rf custom_op

    # 生成自定义算子工程样例
    JSON_NAME=add_custom
    CAMEL_JSON_NAME=`echo $JSON_NAME | sed -r 's/(^|-|_)(\w)/\U\2/g'`
    msopgen gen -i op_dev/${JSON_NAME}.json -f tf -c ai_core-${SOC_SHORT_VERSION} -lan cpp -out ./custom_op
    if [ $? -ne 0 ]; then
        echo "ERROR: msopgen custom op sample failed!"
        return 1
    fi
    echo "INFO: msopgen custom op sample success!"

    cp -rf op_dev/* custom_op
    if [ $? -ne 0 ]; then
        echo "ERROR: copy custom op files failed!"
        return 1
    fi
    if [[ $IS_DYNAMIC != 1 ]]; then
        if [[ $REPLAY_MODE = "batch" ]]; then
            sed -i "s/set(BATCH_MODE_REPLAY_LIST/set(BATCH_MODE_REPLAY_LIST ${CAMEL_JSON_NAME}/g" `grep "set(BATCH_MODE_REPLAY_LIST" -rl custom_op/op_kernel/CMakeLists.txt`
        elif [[ $REPLAY_MODE = "iterator" ]]; then
            sed -i "s/set(ITERATOR_MODE_REPLAY_LIST/set(ITERATOR_MODE_REPLAY_LIST ${CAMEL_JSON_NAME}/g" `grep "set(ITERATOR_MODE_REPLAY_LIST" -rl custom_op/op_kernel/CMakeLists.txt`
        fi
    fi
    sed -i "s#/usr/local/Ascend/latest#$ASCEND_HOME_DIR#g" `grep "/usr/local/Ascend/latest" -rl custom_op/CMakePresets.json`

    # 构建自定义算子包并安装
    bash custom_op/run.sh
    if [ $? -ne 0 ]; then
        echo "ERROR: build and install custom op run package failed!"
        return 1
    fi
    echo "INFO: build and install custom op run package success!"

    # PTA源码仓，可以自行放置zip包
    if [ ! -f "v${PYTORCH_VERSION}.zip" ]; then
        wget https://gitee.com/ascend/pytorch/repository/archive/v${PYTORCH_VERSION}.zip --no-check-certificate
    fi
    rm -rf ${PTA_DIR}; unzip -o -q v${PYTORCH_VERSION}.zip

    # PTA自定义算子注册
    FUNCTION_REGISTE_FIELD=`cat pytorch_patch/npu_native_functions.yaml`
    FUNCTION_REGISTE_FILE="${PTA_DIR}/torch_npu/csrc/aten/npu_native_functions.yaml"
    if ! grep -q "\  $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE; then
        sed -i "/custom:/a \  $FUNCTION_REGISTE_FIELD" $FUNCTION_REGISTE_FILE
    fi
    # PTA自定义算子适配文件
    cp -rf pytorch_patch/*.cpp ${PTA_DIR}/torch_npu/csrc/aten/ops/op_api

    # 编译PTA插件并安装
    (cd ${PTA_DIR}; bash ci/build.sh --python=${PYTHON_VERSION}; pip3 install dist/*.whl --force-reinstall)

    # 执行测试文件
    export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
    python3 test_ops_custom.py
    if [ $? -ne 0 ]; then
        echo "ERROR: run custom op failed!"
        return 1
    fi

    # 解析dump文件为numpy文件
    files=$(ls ./prof_total)
    cd $CURRENT_DIR/prof_total/$files
    msprof --export=on --output=$CURRENT_DIR/prof_total/$files
    if [[ $? -eq 0 ]];then
        echo "INFO: parse success"
    else
        echo "ERROR: pasrse failed"
        return 1
    fi

    # 校验summary文件夹
    summary_list=(
        acl_0_1.csv
        acl_statistic_0_1.csv
        ge_op_execute_0_1.csv
        op_statistic_0_1.csv
        op_summary_0_1.csv
        prof_rule_0.json
        runtime_api_0_1.csv
        task_time_0_1.csv
    )
    if [ $(ls ./device_*/summary/ | wc -l) -eq ${#summary_list[@]} ];then
        for summary in ${summary_list[@]}; do
            if [ ! -f $(pwd)/device_0/summary/$summary ];then
                echo "ERROR: summary files not exist"
                return 1
            fi
        done
        echo "INFO: All summary result exist"
    else
        echo "ERROR: check summary result fail"
        return 1
    fi

    # 校验timeline文件夹
    timeline_list=(
        acl_0_1.json
        ge_op_execute_0_1.json
        msprof_0_1.json
        runtime_api_0_1.json
        task_time_0_1.json
        thread_group_0_1.json
    )
    if [ $(ls ./device_*/timeline/ | wc -l) -eq ${#timeline_list[@]} ];then
        for timeline in ${timeline_list[@]}; do
            if [ ! -f $(pwd)/device_0/timeline/$timeline ];then
                echo "ERROR: timeline files not exist"
                return 1
            fi
        done
        echo "INFO: timeline files exist"
    else
       echo "ERROR: timeline files not exist"
       return 1
    fi
    echo "INFO: Ascend C Add Custom SUCCESS"
}

check_soc_version
main
