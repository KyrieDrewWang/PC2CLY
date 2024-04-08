#!/bin/bash
#SBATCH --job-name=visualizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wangcheng@idea.edu.cn
#SBATCH --partition=vanke-preempted
python visualizer.py --logdir=results/Point2Cyl --model_id=55838_a1513314_0000_1 --dump_dir=dump_55838_a1513314_0000_1 --output_dir=output_55838_a1513314_0000_1 --data_dir=data/
