#!/bin/bash
# 一键运行所有实验
# 使用方法: bash scripts/run_all_exps.sh

set -e

# 创建日志目录
mkdir -p logs

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}   ACT 优化实验 - 批量训练${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "${BLUE}实验配置:${NC}"
echo "  A: 余弦 LR 优化版 (GPU 0) - 增强预热 + 相对角度"
echo "  B: 轻量 Label Smoothing (GPU 1) - 0.02 smoothing"
echo "  C: 修正分段训练 (GPU 2) - 降低 LR + 增加 dropout"
echo "  D: 组合最优配置 (GPU 3) - 全部最佳实践"
echo ""

# 选择运行模式
echo -e "${YELLOW}请选择运行模式:${NC}"
echo "  1) 顺序运行 (一个接一个)"
echo "  2) 并行运行 (同时启动所有实验)"
echo "  3) 只运行指定实验"
echo "  4) 查看实验配置对比"
echo "  5) 监控运行状态"
echo ""
read -p "请输入选项 [1-5]: " mode

case $mode in
    1)
        echo -e "${YELLOW}顺序运行所有实验...${NC}"
        echo ""

        echo -e "${GREEN}[1/4] 运行实验 A: 余弦 LR 优化版${NC}"
        bash scripts/train_exp_A.sh
        echo ""

        echo -e "${GREEN}[2/4] 运行实验 B: 轻量 Label Smoothing${NC}"
        bash scripts/train_exp_B.sh
        echo ""

        echo -e "${GREEN}[3/4] 运行实验 C: 修正分段训练${NC}"
        bash scripts/train_exp_C.sh
        echo ""

        echo -e "${GREEN}[4/4] 运行实验 D: 组合最优配置${NC}"
        bash scripts/train_exp_D.sh
        ;;

    2)
        echo -e "${YELLOW}并行运行所有实验...${NC}"
        echo ""

        echo -e "${GREEN}启动实验 A (GPU 0)...${NC}"
        nohup bash scripts/train_exp_A.sh > logs/exp_A.out 2>&1 &
        pid_A=$!
        sleep 5

        echo -e "${GREEN}启动实验 B (GPU 1)...${NC}"
        nohup bash scripts/train_exp_B.sh > logs/exp_B.out 2>&1 &
        pid_B=$!
        sleep 5

        echo -e "${GREEN}启动实验 C (GPU 2)...${NC}"
        nohup bash scripts/train_exp_C.sh > logs/exp_C.out 2>&1 &
        pid_C=$!
        sleep 5

        echo -e "${GREEN}启动实验 D (GPU 3)...${NC}"
        nohup bash scripts/train_exp_D.sh > logs/exp_D.out 2>&1 &
        pid_D=$!

        echo ""
        echo -e "${GREEN}所有实验已启动!${NC}"
        echo "进程 ID:"
        echo "  实验 A: $pid_A"
        echo "  实验 B: $pid_B"
        echo "  实验 C: $pid_C"
        echo "  实验 D: $pid_D"
        echo ""
        echo -e "${BLUE}监控命令:${NC}"
        echo "  tail -f logs/exp_A.out  # 查看实验 A"
        echo "  tail -f logs/exp_*.out  # 查看所有实验"
        echo "  nvidia-smi              # 查看 GPU 使用"
        echo "  bash scripts/run_all_exps.sh  # 选择 5 监控状态"
        ;;

    3)
        echo "可用的实验:"
        echo "  A - 余弦 LR 优化版"
        echo "  B - 轻量 Label Smoothing"
        echo "  C - 修正分段训练"
        echo "  D - 组合最优配置"
        echo ""
        read -p "请输入要运行的实验 (A/B/C/D, 多个用空格分隔): " exps

        for exp in $exps; do
            case $exp in
                A|a)
                    echo -e "${GREEN}运行实验 A...${NC}"
                    bash scripts/train_exp_A.sh
                    ;;
                B|b)
                    echo -e "${GREEN}运行实验 B...${NC}"
                    bash scripts/train_exp_B.sh
                    ;;
                C|c)
                    echo -e "${GREEN}运行实验 C...${NC}"
                    bash scripts/train_exp_C.sh
                    ;;
                D|d)
                    echo -e "${GREEN}运行实验 D...${NC}"
                    bash scripts/train_exp_D.sh
                    ;;
                *)
                    echo -e "${RED}未知实验: $exp${NC}"
                    ;;
            esac
        done
        ;;

    4)
        echo ""
        echo -e "${GREEN}======================================${NC}"
        echo -e "${GREEN}   实验配置对比表${NC}"
        echo -e "${GREEN}======================================${NC}"
        echo ""
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "参数" "实验 A" "实验 B" "实验 C" "实验 D"
        echo "-------------------|-------------|-------------|-------------|-------------"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "学习率" "1e-5" "1e-5" "3e-5" "1e-5"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "LR 调度器" "余弦" "余弦" "余弦" "余弦"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "预热步数" "8000" "5000" "5000" "5000"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "min_lr_ratio" "0.05" "0.1" "0.1" "0.1"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "Label Smoothing" "0" "0.02" "0" "0.01"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "State Dropout" "0.3" "0.3" "0.2" "0.15"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "Dropout" "0.1" "0.1" "0.15" "0.12"
        printf "%-18s | %-11s | %-11s | %-11s | %-11s\n" "相对角度" "✓" "✗" "✗" "✓"
        echo "-------------------|-------------|-------------|-------------|-------------"
        echo ""
        echo -e "${BLUE}各实验特点:${NC}"
        echo "  A: 基于 0225_act_5 (最佳) + 增强预热 + 相对角度"
        echo "  B: 修复 0225_act_4 (LS不稳定) + 降低 smoothing"
        echo "  C: 修复 0225_act_0 (分段失败) + 降低 LR + 增加 dropout"
        echo "  D: 组合所有最佳实践"
        echo ""
        ;;

    5)
        echo -e "${YELLOW}实验运行状态:${NC}"
        echo ""

        # 检查 GPU 使用情况
        echo -e "${BLUE}GPU 状态:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s: %s | 利用率: %s%% | 显存: %s/%s MB\n", $1, $2, $3, $4, $5}'
        echo ""

        # 检查日志文件
        echo -e "${BLUE}实验日志:${NC}"
        for exp in A B C D; do
            log_file="logs/exp_${exp}.log"
            if [ -f "$log_file" ]; then
                last_step=$(grep -oP 'step:\d+K' "$log_file" 2>/dev/null | tail -1 || echo "未开始")
                last_loss=$(grep -oP 'loss:\d+\.\d+' "$log_file" 2>/dev/null | tail -1 || echo "")
                echo "  实验 $exp: $last_step $last_loss"
            else
                echo "  实验 $exp: 未运行"
            fi
        done
        ;;

    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}   完成!${NC}"
echo -e "${GREEN}======================================${NC}"
