#!/bin/bash
#SBATCH --job-name=R37b_find_best
#SBATCH --account=PDE0022
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/training/find_best_%j.log
#SBATCH --error=/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/logs/R37b_dicetopk_5000ep_real/training/find_best_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=girish.17@buckeyemailosu.edu

echo "Find Best — R37b_dicetopk_5000ep_real | Started: $(date)"

module load python/3.10
cd /fs/ess/PDE0022/hari/brats2026/pipelines/nnunet
set +u
source .venv/bin/activate
set -u

export nnUNet_raw="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/raw"
export nnUNet_preprocessed="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/preprocessed/R5_vanilla"
export nnUNet_results="/fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/data/results/R37b_dicetopk_5000ep_real"

VARIANTS_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training/nnUNetTrainer/variants/loss'))")
cp /fs/ess/PDE0022/hari/brats2026/pipelines/nnunet/custom_trainers/nnUNetTrainerDiceTopK10Loss_5000epochs.py "$VARIANTS_DIR/"

nnUNetv2_find_best_configuration 501 -c 3d_fullres \
    -tr nnUNetTrainerDiceTopK10Loss_5000epochs -p nnUNetPlans -f 0 1 2 3 4

if [ $? -eq 0 ]; then
    echo "Find best COMPLETED at $(date)"
else
    echo "ERROR: Find best FAILED!"; exit 1
fi
