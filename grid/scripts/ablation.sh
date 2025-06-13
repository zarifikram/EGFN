ndim=3

for seed in {0..2}
do
    python run_hydra.py ndim=$ndim horizon=16 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=false mutation=false&
    python run_hydra.py ndim=$ndim horizon=16 method=db n_train_steps=2500 replay_sample_size=0 seed=$seed R0=0.00001 prb=false&
    python run_hydra.py ndim=$ndim horizon=16 method=db n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=false&
    python run_hydra.py ndim=$ndim horizon=16 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=true mutation=false&
    python run_hydra.py ndim=$ndim horizon=16 method=db n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=true&
    python run_hydra.py ndim=$ndim horizon=16 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=true mutation=true&
    python run_hydra.py ndim=$ndim horizon=16 method=db_egfn n_train_steps=2500 replay_sample_size=16 seed=$seed R0=0.00001 prb=true mutation=true crossover=false&
    wait
done

