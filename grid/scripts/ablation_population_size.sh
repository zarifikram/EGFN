for seed in {0..2}
do
    python run_hydra.py ndim=5 horizon=20 method=fm_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=5 num_elites=1&
    python run_hydra.py ndim=5 horizon=20 method=fm_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=10 num_elites=2&
    python run_hydra.py ndim=5 horizon=20 method=fm_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=15 num_elites=3&
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=5 num_elites=1&
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=10 num_elites=2&
    python run_hydra.py ndim=5 horizon=20 method=tb_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=15 num_elites=3&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=5 num_elites=1&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=10 num_elites=2&
    python run_hydra.py ndim=5 horizon=20 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 replay_buf_size=1000 population_size=15 num_elites=3&
    wait
done

