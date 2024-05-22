for seed in {0..2}
do
    python run_hydra.py ndim=3 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    python run_hydra.py ndim=3 horizon=20 method=db_egfn feedback=true n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    python run_hydra.py ndim=4 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    python run_hydra.py ndim=4 horizon=20 method=db_egfn feedback=true n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    python run_hydra.py ndim=5 horizon=20 method=db_egfn feedback=true n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 &
    wait
done