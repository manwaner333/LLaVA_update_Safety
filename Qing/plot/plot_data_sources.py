import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # 画出模型的大小和uncertainty metrics 指标之间的关系
# ad_mini = (66.18 + 53.46 + 54.07 + 50.68)/4
# ad_v15 = (69.32 + 49.70 + 78.97 + 45.97)/4
# ad_v16 = (76.87 + 52.47 + 56.91 + 65.60)/4
# ad_mis = (84.55 + 63.18 + 81.16 + 66.70)/4
#
# ra_mini = (75.84 + 53.03 + 60.90 + 50.65)/4
# ra_v15 = (63.93 + 56.00 + 82.56 + 48.21)/4
# ra_v16 = (80.96 + 50.90 + 67.18 + 66.08)/4
# ra_mis = (86.27 + 63.70 + 83.35 + 68.77)/4
#
# po_mini = (68.77 + 53.54 + 54.41 + 50.71)/4
# po_v15 = (68.72 + 48.15 + 81.34 + 43.12)/4
# po_v16 = (80.48 + 50.64 + 76.52 + 64.44)/4
# po_mis = (86.87 + 63.05 + 84.20 + 65.58)/4
#
#
# gq_mini = (56.59 + 51.13 + 50.83 + 50.25)/4
# gq_v15 = (65.73 + 51.31 + 67.61 + 52.07)/4
# gq_v16 = (64.08 + 52.29 + 60.58 + 52.22)/4
# gq_mis = (69.75 + 57.76 + 68.38 + 58.38)/4
#
#
# hal_mini = (30.52 + 25.92 + 33.45 + 25.96)/4
# hal_v15 = (38.06 + 32.65 + 31.90 + 29.89)/4
# hal_v16 = (37.84 + 42.12 + 33.47 + 40.18)/4
# hal_mis = (36.82 + 39.18 + 32.78 + 40.85)/4
#
# sel_mini = (34.16 + 30.81 + 33.97 + 30.53)/4
# sel_v15 = (41.04 + 38.40 + 38.90 + 36.44 )/4
# sel_v16 = (51.17 + 49.42 + 45.28 + 53.16)/4
# sel_mis = (48.38 + 48.32 + 44.34 + 47.72)/4
#
# models = ['M-Hal', 'Self-Gen', 'POPE-Adversarial', 'POPE-Random', 'POPE-Popular', 'GQA']
# vanilla = [hal_mini, sel_mini, ad_mini, ra_mini, po_mini, gq_mini]
# freeCheck = [hal_v15, sel_v15, ad_v15, ra_v15, po_v15, gq_v15]
# wo_AOP = [hal_v16, sel_v16, ad_v16, ra_v16, po_v16, gq_v16]
# wo_LCL = [hal_mis, sel_mis, ad_mis, ra_mis, po_mis, gq_mis]
# # logicCheckGPT = [50, 78, 85, 90]
#
# # Number of groups
# n_groups = len(models)
#
# # Create the bar plot
# fig, ax = plt.subplots(figsize=(10, 6))
#
# index = np.arange(n_groups)
# bar_width = 0.15
# opacity = 0.8
#
# # Plot each set of bars for the different methods
# rects1 = ax.bar(index, vanilla, bar_width, alpha=opacity, color='deepskyblue', label='MiniGPT-v2')
# rects2 = ax.bar(index + bar_width, freeCheck, bar_width, alpha=opacity, color='aqua', label='LLaVA-1.5-7b')
# rects3 = ax.bar(index + 2*bar_width, wo_AOP, bar_width, alpha=opacity, color='lime', label='LLaVA-1.6-7b')
# rects4 = ax.bar(index + 3*bar_width, wo_LCL, bar_width, alpha=opacity, color='cornflowerblue', label='LLaVA-1.6-mistral')
# # rects5 = ax.bar(index + 4*bar_width, logicCheckGPT, bar_width, alpha=opacity, color='y', label='LogicCheckGPT')
#
# # Add some text for labels, title, and axes ticks
# ax.set_xlabel('Data')
# ax.set_ylabel('Ave_Uncertainty')
# # ax.set_title('Accuracy by model and method')
# ax.set_xticks(index + 2*bar_width)
# ax.set_xticklabels(models)
# # ax.legend()
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=5, fontsize=12)
# # Display the plot
# # plt.show()
#
# # fig.savefig("average_uncertainty_models_up", bbox_inches='tight', pad_inches=0.5)



# # 计算probing高于基准线的程度
#
# m_hal = ((68.5 - 29.29) + (71.4 - 29.29) + (71.8 - 29.29) + (69.9 - 29.29))/4
# self_data = ((58.7 - 35.71) + (59.3 - 35.71) + (59.9 - 35.71) + (58.1 - 35.71))/4
# # print(m_hal, self_data)


"""
import matplotlib.pyplot as plt
import numpy as np

# Number of variables we're plotting.
num_vars = 9  # Example for 6 variables

# Create a 2D array of shape (6,) with your values.
# Here we have 3 observations (LiLMaA 2-78, LiLMaA 2-138, LiLMaA 2-78)
# Replace these values with your actual data
values = np.array([
    [86.82, 91.80, 92.04, 95.89, 96.23, 96.99, 92.73, 93.57, 95.06],  # First observation
    [76.81, 71.72, 73.43, 87.47, 82.43, 84.84, 82.12, 75.74, 78.41],  # Second observation
    [92.35, 93.32, 93.50, 97.86, 98.68, 98.33, 97.85, 98.11, 98.50],
    [81.22, 73.72, 78.06, 92.80, 88.50, 90.05, 94.03, 91.39, 92.79]
])

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made circular, so we need to "complete the loop"
# and append the start to the end.
values = np.concatenate((values, values[:,[0]]), axis=1)
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], ['V1.5-7b \n Adv', 'V1.6-7b \n Adv', 'V1.6-Mis \n Adv', 'V1.5-7b \n Ran', 'V1.6-7b \n Ran', 'V1.6-Mis \n Ran', 'V1.5-7b \n Pop', 'V1.6-7b \n Pop', 'V1.6-Mis \n Pop']
           , size=12, fontweight='bold')

# Draw ylabels
ax.set_rlabel_position(20)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", '100'], color="grey", size=12, fontweight='bold')hao
plt.ylim(0, 100)

# Plot each individual = each line of the data
# I'm assuming you have 3 lines/observations.
# ax.plot(angles, values[0], color='moccasin', linewidth=1.2, linestyle='solid', label='Clear Image')
# ax.fill(angles, values[0], color='moccasin', alpha=0.25)
#
# ax.plot(angles, values[1], color='salmon', linewidth=1.2, linestyle='solid', label='Blurred Image')
# ax.fill(angles, values[1], color='salmon', alpha=0.25)


ax.plot(angles, values[2], color='plum', linewidth=1.2, linestyle='solid', label='Clear Image')
ax.fill(angles, values[2], color='plum', alpha=0.25)

ax.plot(angles, values[3], color='palegreen', linewidth=1.2, linestyle='solid', label='Blurred Image')
ax.fill(angles, values[3], color='palegreen', alpha=0.25)



# ax.plot(angles, values[2], color='green', linewidth=2, linestyle='solid', label='LiLMaA 2-78')
# ax.fill(angles, values[2], color='green', alpha=0.25)

# Add a legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fancybox=True, shadow=True, ncol=2, prop={'size': 12, 'weight': 'bold'})
# plt.show()
plt.tight_layout()
plt.show()
# save_path = "logp_figure_noise.png "
save_path = "probing_figure_noise.png"
# plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0.1,)
"""



"""
import matplotlib.pyplot as plt
import numpy as np

# Number of variables we're plotting.
num_vars = 12 # Example for 6 variables

# Create a 2D array of shape (6,) with your values.
# Here we have 3 observations (LiLMaA 2-78, LiLMaA 2-138, LiLMaA 2-78)
# Replace these values with your actual data
values = np.array([
    [81.70, 85.10, 81.74, 65.06, 54.27, 53.51, 51.83, 57.93, 58.53, 51.95, 57.65, 56.56],  # First observation
    [92.04, 96.99, 95.06, 79.15, 91.80, 96.23, 93.57, 80.31, 86.82, 95.89, 92.73, 78.45],  # Second observation
])

# Compute angle for each axis
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made circular, so we need to "complete the loop"
# and append the start to the end.
values = np.concatenate((values, values[:,[0]]), axis=1)
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], ['V1.6-Mis \n Adv', 'V1.6-Mis \n Ran', 'V1.6-Mis \n Pop', 'V1.6-Mis \n GQA', 'V1.6-7b \n Adv', 'V1.6-7b \n Ran', 'V1.6-7b \n Pop', 'V1.6-7b \n GQA'
    , 'V1.5-7b \n Adv', 'V1.5-7b \n Ran', 'V1.5-7b \n Pop', 'V1.5-7b \n GQA',]
           , size=12, fontweight='bold')

# Draw ylabels
ax.set_rlabel_position(20)
plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", '100'], color="grey", size=12, fontweight='bold')
plt.ylim(0, 100)

# Plot each individual = each line of the data
# I'm assuming you have 3 lines/observations.
ax.plot(angles, values[0], color='palegreen', linewidth=1.2, linestyle='solid', label='without period')
ax.fill(angles, values[0], color='palegreen', alpha=0.25)

ax.plot(angles, values[1], color='plum', linewidth=1.2, linestyle='solid', label='with period')
ax.fill(angles, values[1], color='plum', alpha=0.25)



# ax.plot(angles, values[2], color='green', linewidth=2, linestyle='solid', label='LiLMaA 2-78')
# ax.fill(angles, values[2], color='green', alpha=0.25)

# Add a legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.06), fancybox=True, shadow=True, ncol=2, prop={'size': 12, 'weight': 'bold'})
# plt.show()
plt.tight_layout()
# plt.show()
# save_path = "logp_figure_noise.png "
# save_path = "probing_figure_noise.png"
# save_path = "nli_figure_noise.png"
save_path = 'with_without_period.png'
plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0.1,)

"""



models = ['Avg(-logp)', 'Avg(H)', 'Max(-logp)', 'Max(H)', 'QA', 'BERTScore', 'Unigram', 'NLI', 'Probing', ]
vanilla = [37.22 - 29.29, 32.18 - 29.29, 32.05 - 29.29,  29.71 - 29.29, 40.27 - 29.29, 39.13 - 29.29, 38.68 - 29.29, 44.09 - 29.29, 61.23 - 29.29]
freeCheck = [42.35 - 35.71, 37.18 - 35.71, 41.89 - 35.71, 35.88 - 35.71, 54.72 - 35.71, 42.37 - 35.71, 39.06 - 35.71, 50.80 - 35.71, 63.85 - 35.71]


# Number of groups
n_groups = 9

# Create the bar plot
fig, ax = plt.subplots(figsize=(10, 6))

index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

# Plot each set of bars for the different methods
rects1 = ax.bar(index, vanilla, bar_width, alpha=opacity, color='darkorange', label='M-Hal')
rects2 = ax.bar(index + bar_width, freeCheck, bar_width, alpha=opacity, color='lightseagreen', label='NLI')
# rects3 = ax.bar(index + 2*bar_width, wo_AOP, bar_width, alpha=opacity, color='lime', label='LLaVA-1.6-7b')
# rects4 = ax.bar(index + 3*bar_width, wo_LCL, bar_width, alpha=opacity, color='cornflowerblue', label='LLaVA-1.6-mistral')
# rects5 = ax.bar(index + 4*bar_width, logicCheckGPT, bar_width, alpha=opacity, color='y', label='LogicCheckGPT')

# Add some text for labels, title, and axes ticks
ax.set_xlabel('Methods', fontdict={'fontsize': 16, 'fontweight': 'bold'})
ax.set_ylabel('$\Delta$ AUC-PR', fontdict={'fontsize': 16, 'fontweight': 'bold'})
# ax.set_title('Accuracy by model and method')
ax.set_xticks(index + 0.5*bar_width)
ax.set_xticklabels(models, fontweight='bold', rotation=45)
ax.set_yticklabels(ax.get_yticks(), fontweight='bold')
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# ax.legend()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=5, prop={'size': 14, 'weight': 'bold'})
# Display the plot
plt.tight_layout()
# plt.show()


fig.savefig("data_sources.png", bbox_inches='tight', pad_inches=0.5)
