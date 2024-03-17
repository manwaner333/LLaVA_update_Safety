import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

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



# 计算probing高于基准线的程度

m_hal = ((68.5 - 29.29) + (71.4 - 29.29) + (71.8 - 29.29) + (69.9 - 29.29))/4
self_data = ((58.7 - 35.71) + (59.3 - 35.71) + (59.9 - 35.71) + (58.1 - 35.71))/4
# print(m_hal, self_data)


# 对hidden states 的聚类情况进行分析
file_path = 'result/coco2014_val/llava_v16_mistral_7b_answer_coco_pope_random_extract_info_without_role.bin'
with open(file_path, "rb") as f:
    responses = pickle.load(f)

hidden_states_list = []
labels_list = []

for idx, response in responses.items():
    # Generate a synthetic high-dimensional dataset
    hidden_states = response['logprobs']['combined_hidden_states']
    labels = response['labels']
    len_sentences = len(response['sentences'])
    for i in range(len_sentences):
        hidden_state = hidden_states[i][0]
        label = labels[i]
        if label == 'ACCURATE' or label == 'ANALYSIS':
            true_score = 1.0
        elif label == 'INACCURATE':
            true_score = 0.0
        hidden_states_list.append(hidden_state)
        labels_list.append(true_score)

# # Applying PCA to reduce dimensions to 2
# pca = PCA(n_components=3)
# X_pca_3d = pca.fit_transform(hidden_states_list)
#
# # Applying K-Means clustering
# kmeans = KMeans(n_clusters=2, random_state=0)
# clusters = kmeans.fit_predict(X_pca_3d)
#
# # Plotting in 3D
# fig = plt.figure(figsize=(14, 8))
#
# # Plotting the PCA-reduced data colored by true labels
# ax1 = fig.add_subplot(111, projection='3d')
# for i in range(len(X_pca_3d)):
#     if labels_list[i] == 1:
#         s1 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
#     else:
#         s2 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
# ax1.set_title('True Labels in 3D PCA-reduced Space')
# ax1.set_xlabel('PC1')
# ax1.set_ylabel('PC2')
# ax1.set_zlabel('PC3')
# ax1.legend((s1, s2), ("True", "False"), loc='best')

# # Plotting the PCA-reduced data colored by cluster labels
# ax2 = fig.add_subplot(122, projection='3d')
# for i in range(len(X_pca_3d)):
#     if clusters[i] == 1:
#         o1 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='r', alpha=0.7)
#     else:
#         o2 = ax2.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], X_pca_3d[i, 2], color='y', alpha=0.7)
# ax2.set_title('K-Means Clusters in 3D PCA-reduced Space')
# ax2.set_xlabel('PC1')
# ax2.set_ylabel('PC2')
# ax2.set_zlabel('PC3')
# ax2.legend((o1, o2), ("Class 1", "Class 2"), loc='best')
# plt.tight_layout()
# # path = 'gqa/' + figure_name + '_' + layer + '.png'
# # # os.makedirs(os.path.dirname(path), exist_ok=True)
# # plt.savefig(path, dpi=150, format='png')
# plt.show()



# # Applying PCA to reduce dimensions to 2
# pca = PCA(n_components=2)
# X_pca_3d = pca.fit_transform(hidden_states_list)
#
# # Applying K-Means clustering
# kmeans = KMeans(n_clusters=2, random_state=0)
# clusters = kmeans.fit_predict(X_pca_3d)
#
# # Plotting in 3D
# # fig = plt.figure(figsize=(14, 8))

# Plotting the PCA-reduced data colored by true labels
# ax1 = fig.add_subplot(111)
# for i in range(len(X_pca_3d)):
#     if labels_list[i] == 1:
#         s1 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], color='r', alpha=0.7)
#     else:
#         s2 = ax1.scatter(X_pca_3d[i, 0], X_pca_3d[i, 1], color='y', alpha=0.7)
# ax1.set_title('True Labels in 3D PCA-reduced Space')
# ax1.set_xlabel('PC1')
# ax1.set_ylabel('PC2')
# # ax1.set_zlabel('PC3')
# ax1.legend((s1, s2), ("True", "False"), loc='best')
#
# plt.tight_layout()
# plt.show()



tsne = TSNE(n_components=3, perplexity=40, n_iter=1500)
X_tsne = tsne.fit_transform(np.array(hidden_states_list))

kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(X_tsne)

# Plotting in 3D
fig = plt.figure(figsize=(14, 8))

# Plotting the PCA-reduced data colored by true labels
ax1 = fig.add_subplot(121, projection='3d')
for i in range(len(X_tsne)):
    if labels_list[i] == 1:
        s1 = ax1.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='r', alpha=0.7)
    else:
        s2 = ax1.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
ax1.set_title('True Labels in 3D PCA-reduced Space')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')
ax1.legend((s1, s2), ("True", "False"), loc='best')

ax2 = fig.add_subplot(122, projection='3d')
for i in range(len(X_tsne)):
    if clusters[i] == 1:
        o1 = ax2.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='r', alpha=0.7)
    else:
        o2 = ax2.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
ax2.set_title('K-Means Clusters in 3D PCA-reduced Space')
ax2.set_xlabel('PC1')
ax2.set_ylabel('PC2')
ax2.set_zlabel('PC3')
ax2.legend((o1, o2), ("Class 1", "Class 2"), loc='best')


plt.tight_layout()
plt.show()













