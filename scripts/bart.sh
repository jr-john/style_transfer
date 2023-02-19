#!/bin/bash
#SBATCH -A research
#SBATCH -c 18
#SBATCH --gres gpu:2
#SBATCH --mem-per-cpu 2G
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/textbox_bart.log
#SBATCH --mail-user jerrin.thomas@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name bart_textbox

echo "Loading Modules"
module load cudnn/7.6.5-cuda-10.2
export CUDA_VISIBLE_DEVICES=0,1

echo "Setting Up Environment"
cd /scratch
[ -d "jerrin.thomas" ] || mkdir jerrin.thomas
chmod 700 jerrin.thomas
cd jerrin.thomas
rm -rf ./*

git clone https://github.com/RUCAIBox/TextBox
cd TextBox
bash install.sh
echo "Done Install"

scp jerrin.thomas@ada:/share1/jerrin.thomas/gyafc_em.ckpt textbox/evaluator/utils
scp jerrin.thomas@ada:/share1/jerrin.thomas/gyafc_fr.ckpt textbox/evaluator/utils
echo "Done Importing Checkpoints"

cd dataset
git lfs install
git clone https://huggingface.co/datasets/RUCAIBox/Style-Transfer
mv Style-Transfer/gyafc_em.tgz .
mv Style-Transfer/gyafc_fr.tgz .
tar -zxf gyafc_em.tgz
tar -zxf gyafc_fr.tgz
cd ..
echo "Done Getting Dataset"


python3 run_textbox.py --model=BART --dataset=gyafc_em --model_path=facebook/bart-base
echo "Done EM!"

python3 run_textbox.py --model=BART --dataset=gyafc_fr --model_path=facebook/bart-base
echo "Done FR!"

scp -r saved jerrin.thomas@ada:/share1/jerrin.thomas/bart/