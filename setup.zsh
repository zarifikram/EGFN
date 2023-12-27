conda create -n e-gfn python=3.9
conda activate e-gfn
pip install numpy matplotlib torch scipy tqdm fastrand
pip install hydra-core==1.1
pip install wandb
pip install torch_scatter, torch_geometric, torch_sparse
pip install rdkit