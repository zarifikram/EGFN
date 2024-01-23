for seed in {0..2}
do
    python run_hydra.py ndim=5 horizon=16 method=fm n_train_steps=2500 replay_sample_size=0 seed=$seed R0=0.0001&
    python run_hydra.py ndim=5 horizon=16 method=fm_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed  R0=0.0001&
    python run_hydra.py ndim=5 horizon=16 method=tb n_train_steps=2500 replay_sample_size=0 seed=$seed  R0=0.0001&
    python run_hydra.py ndim=5 horizon=16 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.0001&
    python run_hydra.py ndim=5 horizon=16 method=db n_train_steps=2500 replay_sample_size=0 seed=$seed R0=0.0001&
    python run_hydra.py ndim=5 horizon=16 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.0001&
    wait
done
