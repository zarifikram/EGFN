# python run_hydra.py ndim=3 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.00001 &
python run_hydra.py ndim=4 horizon=20 method=tb augmented=true ri=0.03 n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.00001 &
# python run_hydra.py ndim=5 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=0 R0=0.00001 &
# python run_hydra.py ndim=3 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=1 R0=0.00001 &
python run_hydra.py ndim=4 horizon=20 method=tb augmented=true ri=0.03 n_train_steps=2500 replay_sample_size=16 seed=1 R0=0.00001 &
# python run_hydra.py ndim=5 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=1 R0=0.00001 &
# python run_hydra.py ndim=3 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=2 R0=0.00001 &
python run_hydra.py ndim=4 horizon=20 method=tb augmented=true ri=0.03 n_train_steps=2500 replay_sample_size=16 seed=2 R0=0.00001 &
# python run_hydra.py ndim=5 horizon=20 method=tb augmented=true ri=0.01 n_train_steps=2500 replay_sample_size=16 seed=2 R0=0.00001 &
wait
