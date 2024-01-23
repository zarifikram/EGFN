import gzip, pickle
mols = pickle.load(gzip.open(f'outputs/good_runs/db_egfn_0/results/100_sampled_mols.pkl.gz'))

_, blocks, a, b = mols[0]
print(blocks.smiles)