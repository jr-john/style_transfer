#!/bin/bash
#SBATCH -A research
#SBATCH -c 9
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 2G
#SBATCH --time 4-00:00:00
#SBATCH --output job-logs/bert_em.log
#SBATCH --mail-user jerrin.thomas@research.iiit.ac.in
#SBATCH --mail-type ALL
#SBATCH --job-name bert_em

echo "Setting Up Environment"
cd /scratch
[ -d "jerrin.thomas" ] || mkdir jerrin.thomas
chmod 700 jerrin.thomas
cd jerrin.thomas
rm -rf ./*

git lfs install
git clone https://huggingface.co/datasets/RUCAIBox/Style-Transfer
mv Style-Transfer/gyafc_em.tgz .
mv Style-Transfer/gyafc_fr.tgz .
tar -zxf gyafc_em.tgz
tar -zxf gyafc_fr.tgz

scp -r jerrin.thomas@ada:/share1/jerrin.thomas/models .

cp ~/projects/nnlg/bert_cls.py .
python bert_cls.py

scp -r models jerrin.thomas@ada:/share1/jerrin.thomas