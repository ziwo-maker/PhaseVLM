#!/bin/bash
#============================================================
# GPU Monitor: 持续监控4张显卡，显存全部 < 阈值时自动启动任务
# 用法: bash gpu_monitor.sh
# 日志: ./gpu_monitor.log
#============================================================

# ---- 配置区域 (按需修改) ----
CHECK_INTERVAL=10                     # 每隔多少秒检查一次 (秒)
GPU_COUNT=4                           # 监控的显卡数量
MEM_THRESHOLD=10                      # 显存阈值 (MiB)，每张卡低于此值才视为空闲
TRIGGER_CMD="bash run_alldataset.sh"  # 触发命令
LOG_FILE="./gpu_monitor.log"
LOCK_FILE="/tmp/.gpu_monitor_lock"    # 防止重复触发的锁文件
# ---- 配置结束 ----

# ---- 颜色定义 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---- 日志函数 ----
log() {
    local TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${CYAN}[${TIMESTAMP}]${NC} $*"
    echo "[${TIMESTAMP}] $*" >> "$LOG_FILE"
}
log_green()  { log "${GREEN}$*${NC}"; }
log_yellow() { log "${YELLOW}$*${NC}"; }
log_red()    { log "${RED}$*${NC}"; }

# ---- 退出处理 ----
cleanup() {
    rm -f "$LOCK_FILE"
    log_red "监控已停止 (收到中断信号)"
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- 检查 nvidia-smi 是否可用 ----
if ! command -v nvidia-smi &> /dev/null; then
    log_red "错误: nvidia-smi 未找到，请确认已安装 NVIDIA 驱动"
    exit 1
fi

# ---- 检查实际显卡数量 ----
ACTUAL_GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$ACTUAL_GPU_COUNT" -lt "$GPU_COUNT" ]; then
    log_red "警告: 配置监控 ${GPU_COUNT} 张卡，但系统仅检测到 ${ACTUAL_GPU_COUNT} 张"
    log_yellow "将监控实际存在的 ${ACTUAL_GPU_COUNT} 张显卡"
    GPU_COUNT=$ACTUAL_GPU_COUNT
fi

# ---- 启动信息 ----
log_green "===== GPU 监控脚本启动 ====="
log_green "监控显卡数量 : ${GPU_COUNT}"
log_green "空闲判断阈值 : 显存 < ${MEM_THRESHOLD} MiB"
log_green "检查间隔    : ${CHECK_INTERVAL} 秒"
log_green "触发命令    : ${TRIGGER_CMD}"
log_green "日志文件    : ${LOG_FILE}"
log_green "============================="

# ---- 主循环 ----
while true; do
    ALL_IDLE=true
    IDLE_INFO=""

    # 一次调用拿到所有 GPU 的信息（高效，避免对每张卡单独调用）
    # 输出示例: "GPU-A100, 200 MiB, 80000 MiB"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null)

    IDX=0
    while IFS= read -r line; do
        [ $IDX -ge $GPU_COUNT ] && break

        # 解析: "NVIDIA A100, 200 MiB, 80000 MiB"
        NAME=$(echo "$line" | awk -F',' '{print $1}' | xargs)
        USED=$(echo "$line" | awk -F',' '{print $2}' | awk '{print $1}' | xargs)  # 纯数字 MiB
        TOTAL=$(echo "$line" | awk -F',' '{print $3}' | awk '{print $1}' | xargs)

        # 容错：如果解析失败，默认为 0
        USED=${USED:-0}
        TOTAL=${TOTAL:-0}

        if [ "$USED" -lt "$MEM_THRESHOLD" ] 2>/dev/null; then
            IDLE_INFO="${IDLE_INFO}  GPU${IDX} [${NAME}]: 空闲 ✓  | 显存 ${USED} MiB / ${TOTAL} MiB\n"
        else
            ALL_IDLE=false
            IDLE_INFO="${IDLE_INFO}  GPU${IDX} [${NAME}]: 占用中  | 显存 ${USED} MiB / ${TOTAL} MiB\n"
        fi

        IDX=$((IDX + 1))
    done <<< "$GPU_INFO"

    # 打印当前状态
    echo ""
    log "--- GPU 状态 (阈值 < ${MEM_THRESHOLD} MiB) ---"
    echo -e "$IDLE_INFO"

    # ---- 判断是否全部空闲 ----
    if [ "$ALL_IDLE" = true ]; then
        if [ -f "$LOCK_FILE" ]; then
            log_yellow "所有 GPU 显存 < ${MEM_THRESHOLD} MiB，但上次任务仍在执行中，跳过触发"
        else
            log_green "✅ 所有 ${GPU_COUNT} 张 GPU 显存均 < ${MEM_THRESHOLD} MiB，准备启动任务..."
            log_green "执行: ${TRIGGER_CMD}"

            # 创建锁文件
            echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$LOCK_FILE"

            # 执行触发命令（前台运行，等待完成）
            eval $TRIGGER_CMD
            EXIT_CODE=$?

            # 任务结束，删除锁文件
            rm -f "$LOCK_FILE"

            if [ $EXIT_CODE -eq 0 ]; then
                log_green "✅ 任务执行完成，退出码: ${EXIT_CODE}"
            else
                log_red "❌ 任务执行异常，退出码: ${EXIT_CODE}"
            fi
        fi
    else
        log_yellow "⏳ 部分 GPU 显存 >= ${MEM_THRESHOLD} MiB，继续监控..."
    fi

    sleep "$CHECK_INTERVAL"
done