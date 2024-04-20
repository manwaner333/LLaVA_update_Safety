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


# # 对hidden states 的聚类情况进行分析
# file_path = 'result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info_update_update_hiddens_layer_14_with_role.bin'
# file_path = 'result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info_update_update_hiddens_layer_4_with_role.bin'
file_path = 'result/coco2014_val/llava_v16_7b_answer_coco_pope_random_extract_info_update_update_hiddens_with_role.bin'
filter_file = 'result/coco2014_val/pope_train.json'

with open(file_path, "rb") as f:
    responses = pickle.load(f)

with open(filter_file, 'r') as f:
    keys_val = json.load(f)

hidden_states_list = []
labels_list = []

for idx, response in responses.items():

    # if idx not in keys_val:
    #     continue

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


# # scaler = MinMaxScaler()
# scaler = StandardScaler()
# normalized_data = scaler.fit_transform(np.array(hidden_states_list))
#
# # Applying PCA to reduce dimensions to 2
# pca = PCA(n_components=2)
# X_pca_3d = pca.fit_transform(normalized_data)
#
# # Applying K-Means clustering
# kmeans = KMeans(n_clusters=2, random_state=0)
# clusters = kmeans.fit_predict(X_pca_3d)
#
# # Plotting in 3D
# fig = plt.figure(figsize=(14, 8))
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



# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(np.array(hidden_states_list))
# normalized_data = np.maximum(0, np.array(hidden_states_list))
normalized_data = 1 / (1 + np.exp(-np.array(hidden_states_list)))
# 降维度
tsne = TSNE(n_components=3, perplexity=10, n_iter=2000)
X_tsne = tsne.fit_transform(normalized_data)

# pca = PCA(n_components=3)
# X_tsne = pca.fit_transform(normalized_data)
# X_tsne[:, 0] = X_tsne[:, 0] * 0.3
# X_tsne[:, 1] = X_tsne[:, 1] * 3

# 聚类处理
# kmeans = KMeans(n_clusters=2, random_state=0)
# clusters = kmeans.fit_predict(X_tsne)

dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_tsne)

# Plotting in 3D
fig = plt.figure(figsize=(5.8, 5.2))
# Plotting the PCA-reduced data colored by true labels
ax1 = fig.add_subplot(111, projection='3d')
for i in range(len(X_tsne)):
    if labels_list[i] == 1:
        s1 = ax1.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='r', alpha=0.7)
    else:
        s2 = ax1.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
ax1.view_init(azim=30)
# ax1.set_title('Layer=16 in 3D PCA-reduced Space')
# ax1.set_xlabel('PC1', fontdict={'fontsize': 10, 'fontweight': 'bold'})
# ax1.set_ylabel('PC2', fontdict={'fontsize': 10, 'fontweight': 'bold'})
# ax1.set_zlabel('PC3', fontdict={'fontsize': 10, 'fontweight': 'bold'})
ax1.set_xlim(-40, 20)
ax1.set_ylim(-40, 30)
ax1.set_zlim(-40, 20)
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='z', labelsize=12)

ax1.legend((s1, s2), ("True", "False"), loc='best', prop={'size': 10, 'weight': 'bold'})
plt.tight_layout()
save_path = "Qing/plot/embedding_reduced_space_32.png"
plt.show()

# plt.savefig(save_path, dpi=300, format='png', bbox_inches='tight', pad_inches=0.1)



# fig = plt.figure(figsize=(6, 6))
# ax2 = fig.add_subplot(111, projection='3d')
# for i in range(len(X_tsne)):
#     if clusters[i] == 1:
#         o1 = ax2.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
#     else:
#         o2 = ax2.scatter(X_tsne[i, 0], X_tsne[i, 1], X_tsne[i, 2], color='r', alpha=0.7)
# ax2.view_init(azim=30)
# # ax2.set_title('Layer=32 in 3D PCA-reduced Space')
# ax2.set_xlabel('PC1')
# ax2.set_ylabel('PC2')
# ax2.set_zlabel('PC3')
# ax2.legend((o1, o2), ("Class 1", "Class 2"), loc='best')

# plt.tight_layout()
# plt.show()
# save_path = "Qing/embedding_reduced_space.png "
# plt.savefig(save_path, dpi=150, format='png')


#
# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(111)
# for i in range(len(X_tsne)):
#     if labels_list[i] == 1:
#         s1 = ax1.scatter(X_tsne[i, 1], X_tsne[i, 2], color='r', alpha=0.7)
#     else:
#         s2 = ax1.scatter(X_tsne[i, 1], X_tsne[i, 2], color='y', alpha=0.7)
# ax1.set_title('True Labels in 2D PCA-reduced Space')
# ax1.set_xlabel('PC1')
# ax1.set_ylabel('PC2')
# ax1.legend((s1, s2), ("True", "False"), loc='best')
# plt.xlim(-30, 30)  # 设置x轴范围
# plt.ylim(-10, 10)  # 设置y轴范围
#
# plt.tight_layout()
# plt.show()


"""
scaler = MinMaxScaler()
# scaler = StandardScaler()
normalized_data = scaler.fit_transform(np.array(hidden_states_list))

# Applying PCA to reduce dimensions to 2
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(normalized_data)
X_pca_2d[:, 0] = X_pca_2d[:, 0] * 0.3
X_pca_2d[:, 1] = X_pca_2d[:, 1] * 3

# Applying K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
clusters = kmeans.fit_predict(X_pca_2d)

fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(111)
for i in range(len(X_pca_2d)):
    if labels_list[i] == 1:
        s1 = ax1.scatter(X_pca_2d[i, 1], X_pca_2d[i, 0], color='r', alpha=0.7)
    else:
        s2 = ax1.scatter(X_pca_2d[i, 1], X_pca_2d[i, 0], color='y', alpha=0.7)
ax1.set_title('True Labels in 2D PCA-reduced Space')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.legend((s1, s2), ("True", "False"), loc='best')
plt.xlim(-100, 100)  # 设置x轴范围
plt.ylim(-100, 100)  # 设置y轴范围

plt.tight_layout()
plt.show()
"""
