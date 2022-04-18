clean:
	rm ./*.out

train:
	sbatch -o logs slurm/train.slurm