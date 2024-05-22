import pickle, gzip, os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder = "./qm9str_runtime/"
egfn_data = []
seeds = [0, 1, 2]

modes = [
     'TB_PRT_SSR_EGFN_BETA1', 'TB_PRT_SSR_BETA1'
]

labels = ['EGFN(TB)', 'TB']



for i in modes:
    data = []
    for j in seeds:
        with open(f"{folder}{i}_SEED{j}/final_log.pkl", "rb") as f:
            data.append(pickle.load(f))
    egfn_data.append(data)

egfn_times = np.array([egfn_data[0][seed][190]['time'] - egfn_data[0][seed][50]['time'] for seed in seeds])
tb_times = np.array([egfn_data[1][seed][190]['time'] - egfn_data[1][seed][50]['time'] for seed in seeds])

n_epochs = 140
egfn_times = egfn_times / n_epochs
tb_times = tb_times / n_epochs
print(f"EGFN time: {np.mean(egfn_times)}+-{np.std(egfn_times)}")
print(f"TB time: {np.mean(tb_times)}+-{np.std(tb_times)}")