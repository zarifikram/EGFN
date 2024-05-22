for seed in {0..2}
do
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 top_sample_perc=0.2&
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 top_sample_perc=0.5&
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 top_sample_perc=0.8&
    wait
done

