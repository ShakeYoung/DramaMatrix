#!/bin/bash
#SBATCH --job-name=DramaMatrix_Pipeline
#SBATCH --output=logs/dramamatrix_%j.log
#SBATCH --error=logs/dramamatrix_%j.err
#SBATCH --partition=3090         # 对应你之前在 WMG 集群常使用的 3090 节点分区
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8        # 分配足够的 CPU 进行并发的爬虫或剪辑 (FFMPEG) 操作
#SBATCH --mem=32G                # 内存配置
#SBATCH --gres=gpu:1             # 如果后续 Agent 分配本地大模型推理，或是引入本地的 SD/视频生成模型，需要申明 GPU
#SBATCH --time=24:00:00          # 预估最大运行时间

echo "=========================================================="
echo "🚀 Starting DramaMatrix Multi-Agent Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "=========================================================="

# 1. 切换到作业提交时的目录 (通常建议将代码放到集群的 /home 或工作存储区)
# 注意：你在集群上的路径可能需要替换。目前自动挂载当前提交流目录。
cd $SLURM_SUBMIT_DIR

# 如果 logs 文件夹不存在，则在此创建以避免输出报错
mkdir -p logs

# 2. 加载集群的 Anaconda 环境或加载必要的 Module
# module load anaconda3/2023.09
# source activate dramamatrix_env

# 若直接使用我们刚才建立的 venv 环境，请确保集群系统也支持，或者在集群上重新创建:
# 激活虚拟环境
if [ -d "venv" ]; then
    echo "Activating local venv..."
    source venv/bin/activate
else
    echo "⚠️ Warning: venv directory not found! Make sure to install requirements on the cluster."
fi

# 3. 运行主程序的 LangGraph 网络
echo "Running main.py..."
python main.py

echo "=========================================================="
echo "✅ Job completed at: $(date)"
echo "=========================================================="
