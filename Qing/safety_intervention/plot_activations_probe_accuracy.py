import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re



def get_data(file_path):
    res = [[] for i in range(0, 32)]
    with open(file_path, "r") as file:
        for line in file:
            item = json.loads(line)
            id = item['id']
            match = re.match(r'layer_(\d+)_head_(\d+)', id)
            best_accuracy = item['best_accuracy']
            if match:
                layer_num = int(match.group(1))
                head_num = int(match.group(2))
                print(f'Layer: {layer_num}, Head: {head_num}')
            else:
                print('No match found')
            res[layer_num].append(best_accuracy)
    data = np.array(res)
    return data


def plot_3_heatmap(data1, data2, data3):
    # 创建一个包含3个子图的1行布局
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 绘制第一个热力图
    sns.heatmap(data1, ax=axes[0], vmin=0, vmax=1, cmap='YlGnBu', fmt='.2f')
    axes[0].set_title('Instrauction with text (same source as training dataset)')

    # 绘制第二个热力图
    sns.heatmap(data2, ax=axes[1], vmin=0, vmax=1, cmap='YlGnBu', fmt='.2f')
    axes[1].set_title('Instrauction with text (another dataset)')

    # 绘制第三个热力图
    sns.heatmap(data3, ax=axes[2], vmin=0, vmax=1, cmap='YlGnBu', fmt='.2f')
    axes[2].set_title('Instrauction with text and image (another dataset)')

    # 调整布局
    plt.tight_layout()

    # 显示图像
    # plt.show()
    plt.savefig('images/heatmap_case24.png', bbox_inches='tight')


def plot_heatmap(data):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, vmin=0.0, vmax=70.0, cmap='YlGnBu', fmt='.2f', annot=True, cbar_kws={'orientation': 'horizontal'})  # annot=True,
    # plt.title('示例热力图')
    plt.xlabel('Head')
    plt.ylabel('Layer')

    # 调整布局
    plt.tight_layout()

    plt.show()
    # plt.savefig('images/heatmap_pos_neg_cases_regression.png', bbox_inches='tight')


def plot_layer_strength_heatmap(data):
    plt.figure(figsize=(8, 8))
    x_labels = [0.5, 1.0, 1.5, 2.0]
    y_labels = [4, 9, 14, 19, 24, 29, 31]
    sns.heatmap(data, vmin=0.0, vmax=70.0, cmap='YlGnBu', fmt='.2f', xticklabels=x_labels, yticklabels=y_labels, annot=True, cbar_kws={'orientation': 'horizontal'})  # annot=True,
    # plt.title('示例热力图')
    plt.xlabel('Strength')
    plt.ylabel('Layer')

    # 调整布局
    plt.tight_layout()

    plt.show()
    # plt.savefig('images/heatmap_pos_neg_cases_regression.png', bbox_inches='tight')




if __name__ == "__main__":

    # case 1
    # file_path1 = "result/lmalpaca_xstest/llava_v1.5_7b_probe_layer_head_new_new.json"
    # file_path2 = "result/safebench_alpaca/llava_v1.5_7b_probe_layer_head_new.json"
    # file_path3 = "result/safebench_vlguard/llava_v1.5_7b_probe_layer_head_new.json"

    #case 2
    # file_path1 = "result/safebenchalpaca_xstest/llava_v1.5_7b_probe_layer_head_new_new.json"
    # file_path2 = "result/lmalpaca/llava_v1.5_7b_probe_layer_head_on_safebenalpacaxtest.json"
    # file_path3 = "result/safebench_vlguard/llava_v1.5_7b_probe_layer_head_on_safebenalpacaxtest.json"
    # # file_path3 = "result/vlguard/llava_v1.5_7b_probe_layer_head_on_safebenalpacaxtest.json"
    #
    # data1 = get_data(file_path1)
    # data2 = get_data(file_path2)
    # data3 = get_data(file_path3)
    #
    # plot_3_heatmap(data1, data2, data3)

    # 画出利用positive 和 negative cases 得到的激活值， 来进行的probe
    # data_path = "result/vlguard/llava_v1.5_7b_probe_layer_head_on_pos_nes_cases.json"
    # data = get_data(data_path)
    # plot_heatmap(data)

    # 画出每一层以及不同强度的结果情况
    # data = [[59.04, 55.04, 59.61, 55.81], [59.53, 23.24, 22.0, 24.88], [52.87, 32.23, 8.3, 3.78], [59.39, 40.75, 27.92, 3.17], [66.16, 59.36, 19.61, 0.76], [65.78, 56.60, 0.75, 0.76], [64.15, 0.74, 0.75, 0.32]]

    data = [[50.60, 50.40, 50.60, 50.80], [64.38, 63.19, 60.75, 61.23], [49.80, 45.62, 40.44, 38.25], [49.00, 45.62, 37.25, 28.49], [49.80, 41.24, 34.86, 25.90], [46.41, 38.45, 27.09, 13.94], [37.05, 33.47, 24.13, 11.04]]
    plot_layer_strength_heatmap(data)






