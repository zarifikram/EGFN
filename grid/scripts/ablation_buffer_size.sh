for seed in {0..2}
do
    python run_hydra.py ndim=5 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000&
    python run_hydra.py ndim=5 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=5000&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=5000&
    python run_hydra.py ndim=5 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=10000&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=10000&
    wait
done


python run_hydra.py ndim=3 horizon=20 method=db n_train_steps=2500 replay_sample_size=16 R0=0.00001 replay_buf_size=1000
    