config?=config.toml

clean:
	rm ./*.out

train:
	sbatch -o slurm_logs/train-$(config) slurm/train.slurm --config $(config)

predict:
	sbatch -o slurm_logs/predict-$(config) slurm/train.slurm --config $(config)
