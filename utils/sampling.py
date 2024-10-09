from datetime import datetime

import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    
    for i in range(num_users):
        
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def noniid(dataset, args):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set
    
    # # Plotting
    # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    # fig.suptitle("MNIST Subsets for Each Client", fontsize=16)
    # for client_id, ax in enumerate(axes.flatten()):
    # # 从客户端数据集中随机选择一个样本
    #     client_data = list(dict_users[client_id])
    #     sample_idx = np.random.choice(client_data)
    #     img, label = dataset[sample_idx]
        
    #     # 显示图像
    #     ax.imshow(img.squeeze(), cmap='gray')
    #     ax.set_title(f"Client {client_id}\nLabel: {label}")
    #     ax.axis('off')
        
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92)
    # plt.show()
    
    import os
    output_dir = 'client_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制每个客户端的类别分布柱状图
    for client_id in range(rand_num):
        # 获取客户端的标签列表
        client_data = list(dict_users[client_id])
        labels = [dataset[idx][1] for idx in client_data]
        
        # 统计标签数量
        label_counts = Counter(labels)
        
        # 绘制柱状图
        plt.figure(figsize=(8, 4))
        plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
        plt.xlabel('MNIST Class')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of MNIST Classes for Client {client_id}')
        plt.xticks(range(10))
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f'client_{client_id}_distribution.png'))
        plt.close()  # 关闭图像以释放内存


    return dict_users





def noniid_own(dataset, args):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(args.num_users)}
    
    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num+1, size=args.num_users)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    data_by_class = defaultdict(list)
    for idx, (img, label) in enumerate(dataset):
        data_by_class[label].append(idx)
    num_clients = 100
    random_num_size = [600] * num_clients  # 每个客户端分配 100 个样本
    dict_users = {}

    # 为每个客户端分配类别，并分配数据
    #np.random.seed(42)  # 确保结果可重复
    for i in range(num_clients):
        # 随机选择最多3个类别分配给该客户端
        chosen_classes = np.random.choice(list(data_by_class.keys()), 3, replace=False)
        
        # 从所选类别中抽取指定数量的数据
        client_data = []
        for cls in chosen_classes:
            class_data = data_by_class[cls]
            num_samples = random_num_size[i] // len(chosen_classes)
            
            # 如果类别样本不足，则使用该类别的所有样本
            sampled_data = np.random.choice(class_data, num_samples, replace=False)
            client_data.extend(sampled_data)
        
        # 如果由于取整分配有些样本未分配完，则再随机选一部分
        if len(client_data) < random_num_size[i]:
            additional_samples = np.random.choice(client_data, random_num_size[i] - len(client_data), replace=True)
            client_data.extend(additional_samples)

        dict_users[i] = set(client_data)
    
    # # Plotting
    # fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
    # fig.suptitle("MNIST Subsets for Each Client", fontsize=16)
    # for client_id, ax in enumerate(axes.flatten()):
    # # 从客户端数据集中随机选择一个样本
    #     client_data = list(dict_users[client_id])
    #     sample_idx = np.random.choice(client_data)
    #     img, label = dataset[sample_idx]
        
    #     # 显示图像
    #     ax.imshow(img.squeeze(), cmap='gray')
    #     ax.set_title(f"Client {client_id}\nLabel: {label}")
    #     ax.axis('off')
        
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.92)
    # plt.show()
    
    draw(num_clients,dataset,dict_users)


    return dict_users



def draw(num_clients,dataset,dict_users):
    import os
    output_dir = 'client_distributions'
    os.makedirs(output_dir, exist_ok=True)

    # 绘制每个客户端的类别分布柱状图
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
    for client_id in range(num_clients):
        # 获取客户端的标签列表
        client_data = list(dict_users[client_id])
        labels = [dataset[idx][1] for idx in client_data]
        
        # 统计标签数量
        label_counts = Counter(labels)
        
        # 绘制柱状图
        plt.figure(figsize=(8, 4))
        plt.bar(label_counts.keys(), label_counts.values(), color='skyblue')
        plt.xlabel('MNIST Class')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of MNIST Classes for Client {client_id}')
        plt.xticks(range(10))
        
        
        output_dir = os.path.join('draw', timestamp)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图像
        plt.savefig(os.path.join(output_dir, f'client_{client_id}_distribution.png'))
        plt.close()  # 关闭图像以释放内存
