
import matplotlib.pyplot as plt
import numpy as np


'''基于DBSCAN算法的聚类图'''
def plot_cluster_DBSCAN(cluster_data,cluster_label,core_sample):
    '''
    :param cluster_data: 训练数据集
    :param cluster_label: 数据集类别标签（训练后）
    :param core_sample: 核心样本点，属于核心样本点的位置用1标记，其他用0
    :return:
    '''
    unique_labels=set(cluster_label)
    #根据类别数量生成颜色数量
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #去除噪声的类别数量
    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 把噪声涂上黑色
            col = [0, 0, 0, 1]

        class_member_mask = (cluster_label == k)
        # 指定特定类别的点(核心点)
        xy = cluster_data[class_member_mask & core_sample]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8)
        # 指定类别（非核心点）
        xy = cluster_data[class_member_mask & ~core_sample]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()


'''基于kmeans的聚类图'''
def plot_cluster_kmeans(cluster_data,cluster_label):
    '''
    :param cluster_data: 训练数据集
    :param cluster_label: 数据集类别标签（训练后）
    :return:
    '''
    unique_labels=set(cluster_label)
    #根据类别数量生成颜色数量
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    #去除噪声的类别数量
    n_clusters_ = len(unique_labels) - (1 if -1 in unique_labels else 0)
    for k, col in zip(unique_labels, colors):
        class_member_mask = (cluster_label == k)
        xy = cluster_data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=8)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()