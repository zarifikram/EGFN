python run_hydra.py ndim=5 horizon=10 method=db n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.00001 &
# python run_hydra.py ndim=3 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.01 &
python run_hydra.py ndim=5 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.00001 &
# python run_hydra.py ndim=5 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.01 &
wait