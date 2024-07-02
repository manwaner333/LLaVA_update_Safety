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
    sns.heatmap(data, vmin=0.45, vmax=1, cmap='YlGnBu', fmt='.2f', cbar_kws={'orientation': 'horizontal'})  # annot=True,
    # plt.title('示例热力图')
    plt.xlabel('Head')
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
    data_path = "result/vlguard/llava_v1.5_7b_probe_layer_head_on_pos_nes_cases.json"
    data = get_data(data_path)
    plot_heatmap(data)








