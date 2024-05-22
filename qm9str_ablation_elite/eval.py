import pickle, gzip, os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder = "./qm9str_ablation_elite/"
egfn_data = []
seeds = [0, 1, 2]
# modes = [
#     "fm_3", "fm_egfn_3", "db_3", "db_egfn_3", "tb_3", "tb_egfn_3", 
#     "fm_4", "fm_egfn_4", "db_4", "db_egfn_4", "tb_4", "tb_egfn_4", 
#     "fm_5", "fm_egfn_5", "db_5", "db_egfn_5", "tb_5", "tb_egfn_5", 
# ]
modes = [
    'TB_PRT_SSR_EGFN_POP10_EPS2_BETA1', 'TB_PRT_SSR_EGFN_POP10_EPS4_BETA1', 'TB_PRT_SSR_EGFN_POP10_EPS6_BETA1'
]

labels = ['eps = 0.2', 'eps = 0.4', 'eps = 0.6']



for i in modes:
    data = []
    for j in seeds:
        with open(f"{folder}{i}_SEED{j}/final_log.pkl", "rb") as f:
            data.append(pickle.load(f))
    egfn_data.append(data)

# time1 = egfn_data[3][0][1990]['time'] - egfn_data[3][0][10]['time']
# time2 = egfn_data[2][0][1990]['time'] - egfn_data[2][0][10]['time']


# print(f"time 1 is {(time1 - time2) * 100 / time2} percent greater than time 2")

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
for i in range(1*3):
    x = egfn_data[i][1].keys()
    y = []
    for j in range(len(seeds)):
        y.append([egfn_data[i][j][step]['all - Num modes'] for step in x])
    y = np.array(y)
    # calculate the 95% confidence interval
    y_mean = np.mean(y, axis=0)
    yerr = np.std(y, axis=0) * 1.96 / np.sqrt(len(seeds)) /2
    axes[0].plot(x, y_mean, label=labels[i])
    axes[0].fill_between(x, y_mean-yerr, y_mean+yerr, alpha=0.2)


for i in range(1*3):
    x = egfn_data[i][1].keys()
    y = []
    for j in range(len(seeds)):
        y.append([egfn_data[i][j][step]['all - relative mean error to target'] for step in x])
    y = np.array(y)
    # calculate the 95% confidence interval
    y_mean = np.mean(y, axis=0)
    yerr = np.std(y, axis=0) * 1.96 / np.sqrt(len(seeds))/2

    axes[1].plot(x, y_mean, label=labels[i])
    axes[1].fill_between(x, y_mean-yerr, y_mean+yerr, alpha=0.2)

titles = ["Flow Matching", "Trajectory Balance","Detailed Balance"]
xlabel = "Number of States Visited(x10\N{SUPERSCRIPT FOUR})"
for i in range(3):
    axes[0].set_ylabel("Number of Modes")
    axes[1].set_ylabel("Relative Mean Error")
    axes[1].set_xlabel("Number of Training Steps")
    # axes[0].set_title(titles[0])

# Add a legend
for ax in axes:
   
    # ax.legend(loc='upper right')
    ax.set_xlim([0, 2000])
# Adjust layout for better appearance
        
# # add a single legend box for all the subplots
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1),fancybox=False, shadow=False, ncol=4)
# make sure the legend box is not cut off
plt.subplots_adjust(bottom=0.15)
# Common legend for all subplots

# Adjust layout to prevent clipping of legends
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.tight_layout()
plt.show()
