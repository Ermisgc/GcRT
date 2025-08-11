#!/bin/bash
if [ $# -lt 1 ]; then   # $#参数数量 lt:less than
    echo "Use: $0 <executable> [args]"
    echo "Like: $0 ./bin/Debug/GcRT"
    exit 1
fi

executable="$1"
exec_name=$(basename "$executable")
exec_name=${exec_name:0:-4}   # 这里是为了去掉.exe
shift

# 根据程序名称和时间戳设定日志文件路径
timestamp=$(date +"%Y%m%d_%H%M%S")
nvprof_log_path="nvprof_log"
mkdir -p ${nvprof_log_path}
mkdir -p ${nvprof_log_path}/${exec_name}
log_file="./${nvprof_log_path}/${exec_name}/${timestamp}_profile.log"
profile_file="./${nvprof_log_path}/${exec_name}/${timestamp}_profile.nvvp"

NVPROF_OPTIONS=(
    "--log-file" "${log_file}"
    "--normalized-time-unit" "ms"
)

echo "======================================================"
echo "启动 nvprof 分析: $executable"
echo "日志文件: $log_file"
echo "性能文件: $profile_file"
echo "程序参数: $@"  # 这意为剩余的所有参数
echo "======================================================"

nvprof "${NVPROF_OPTIONS[@]}" "$executable" "$@"
exit_code=$?  # 返回值

echo "======================================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ 分析完成！"
    echo "文本报告生成完毕: $log_file"
    # echo "可视化报告生成完毕：$profile_file(可用Nsight Systems打开)"
else
    echo "✗ 执行失败，错误码: $exit_code"
    echo "检查日志文件获取详细信息: $log_file"
fi

echo "======================================================"
exit $exit_code