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
# transfer 能力的验证

models = ['Popular', 'Adversarial', 'Random', 'GQA', 'M-Hal', 'IHAD']
vanilla = [97.99, 91.3, 98.00, 79.6, 85.4, 87.8]
freeCheck = [97.99, 88.57, 95.04, 72.39, 74.76, 84.15]
# wo_AOP = [hal_v16, sel_v16, ad_v16, ra_v16, po_v16, gq_v16]
# wo_LCL = [hal_mis, sel_mis, ad_mis, ra_mis, po_mis, gq_mis]


# Number of groups
n_groups = len(models)

# Create the bar plot
fig, ax = plt.subplots(figsize=(8, 5))

index = np.arange(n_groups)
bar_width = 0.15
opacity = 0.8

# Plot each set of bars for the different methods
rects1 = ax.bar(index, vanilla, bar_width, alpha=opacity, color='darkorange', label='Independent Classifier')
rects2 = ax.bar(index + bar_width, freeCheck, bar_width, alpha=opacity, color='lightseagreen', label='universal classifier')
# rects3 = ax.bar(index + 2*bar_width, wo_AOP, bar_width, alpha=opacity, color='lime', label='LLaVA-1.6-7b')
# rects4 = ax.bar(index + 3*bar_width, wo_LCL, bar_width, alpha=opacity, color='cornflowerblue', label='LLaVA-1.6-mistral')
# rects5 = ax.bar(index + 4*bar_width, logicCheckGPT, bar_width, alpha=opacity, color='y', label='LogicCheckGPT')

# Add some text for labels, title, and axes ticks
ax.set_xlabel('Data', fontdict={'fontsize': 14, 'fontweight': 'bold'})
ax.set_ylabel('AUC-PR', fontdict={'fontsize': 14, 'fontweight': 'bold'})
# ax.set_title('Accuracy by model and method')
ax.set_xticks(index + 2*bar_width)
ax.set_xticklabels(models)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
# ax.legend()
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=5, prop={'size': 14, 'weight': 'bold'})
# Display the plot
plt.show()

# fig.savefiig("different_classifier.png", bbox_inches='tight', pad_inches=0.5)


