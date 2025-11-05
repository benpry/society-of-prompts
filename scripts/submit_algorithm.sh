#!/bin/zsh
#SBATCH --job-name=SoP
#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops,sc-loprio
#SBATCH --output=slurm-output/sop-%j.out
#SBATCH --error=slurm-output/sop-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[ampere|hopper]
#SBATCH --time=12:00:00

source ~/.zshrc
cd ~/society-of-prompts

# install the uv environment if it doesn't exist
if [ ! -d "${SCR_ROOT_DIR}/uv/sop" ]; then
    mkdir -p "${SCR_ROOT_DIR}/uv"
    uv venv ${SCR_ROOT_DIR}/uv/sop
fi

# activate the environment
source ${SCR_ROOT_DIR}/uv/sop/bin/activate
uv sync
uv tool install prime
prime env install primeintellect/livecodebench

zsh scripts/run_vllm_server.sh &
sleep 3m
python src/algorithm.py
wait