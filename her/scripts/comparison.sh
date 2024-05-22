train_steps=2500

for seed in {0..2}
do
    python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 feedback=True seed=$seed &
    python run_hydra.py method=db n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    # python run_hydra.py method=tb_egfn n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    # python run_hydra.py method=tb_egfn n_train_steps=$train_steps replay_sample_size=16 feedback=True seed=$seed &
    # python run_hydra.py method=tb n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    # python run_hydra.py method=db n_train_steps=$train_steps replay_sample_size=0 seed=$seed &
    # python run_hydra.py method=ppo n_train_steps=$train_steps replay_sample_size=0 mbsize=32 seed=$seed &
    python run_hydra.py method=sac n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    # python run_hydra.py method=mars n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    python run_hydra.py method=tb augmented=true n_train_steps=$train_steps replay_sample_size=16 seed=$seed &
    python run_hydra.py method=db_egfn n_train_steps=$train_steps replay_sample_size=16 random_policy=True seed=$seed &
    wait
done
