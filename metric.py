from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.nn.functional import normalize
from scipy.io import savemat

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur


def inference_new(loader, model, device, view, data_size, p_sample,adaptive_weight):
    model.eval()
    cluster_common_feature = []
    labels_vector = []
    for step, (x, y, _) in enumerate(loader):
        for v in range(view):
            x[v]=x[v].to(device)
        with torch.no_grad():
            private_x,low_feature, decoder_feature=model(x)  
            common_feature=model.forward_common_future(low_feature,p_sample,adaptive_weight)
            common_feature = common_feature.detach()
            cluster_common_feature.extend(common_feature.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    cluster_common_feature = np.array(cluster_common_feature)
    return labels_vector, cluster_common_feature


def inference_tsne(loader, model, device, view, data_size, p_sample,adaptive_weight):
    model.eval()
    cluster_common_feature = []
    labels_vector = []
    #新增
    all_private_x = []
    all_common_feature = []
    for step, (x, y, _) in enumerate(loader):
        for v in range(view):
            x[v]=x[v].to(device)
        with torch.no_grad():
            private_x,low_feature, decoder_feature=model(x)
            # if(low_feature[0].shape[0]!=256):
            #     for v in range(view):
            #         padding_rows = 256 - low_feature[v].shape[0]
            #         # 使用零张量作为填充，并在维度0上进行拼接
            #         padding_tensor = torch.zeros(padding_rows, low_feature[v].size(1)).to(device)  # 创建零张量
            #         low_feature[v] = torch.cat((low_feature[v], padding_tensor), dim=0)  #
            print("计算准确率的low_feature的shape",low_feature[v].shape)
            common_feature=model.forward_common_future(low_feature,p_sample,adaptive_weight)
            common_feature = common_feature.detach()
            cluster_common_feature.extend(common_feature.cpu().detach().numpy())
            #新增
            all_private_x.append(private_x[0])
            all_common_feature.append(common_feature)
            

        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    cluster_common_feature = np.array(cluster_common_feature)
    #新增
    all_private_x = torch.cat(all_private_x, dim=0)  # 转换为张量
    all_common_feature = torch.cat(all_common_feature, dim=0)  # 转换为张量

    all_private_x = all_private_x.detach().cpu().numpy()  # 将 x_new 从 GPU 复制到 CPU，并转换为 NumPy 数组
    all_common_feature = all_common_feature.detach().cpu().numpy()  # 将 common_new 从 GPU 复制到 CPU，并转换为 NumPy 数组
    # 将 NumPy 数组保存为 .mat 文件
    data = {'x_new': all_private_x, 'common_new': all_common_feature}
    savemat('Hdigit_epoch1.mat', data)


    return labels_vector, cluster_common_feature



def inference(loader, model, device, view, data_size, p_sample,adaptive_weight):
    model.eval()
    cluster_common_feature = []
    labels_vector = []
    for step, (x, y, _) in enumerate(loader):
        for v in range(view):
            x[v]=x[v].to(device)
        with torch.no_grad():
            private_x,low_feature, decoder_feature=model(x)
            # if(low_feature[0].shape[0]!=256):
            #     for v in range(view):
            #         padding_rows = 256 - low_feature[v].shape[0]
            #         # 使用零张量作为填充，并在维度0上进行拼接
            #         padding_tensor = torch.zeros(padding_rows, low_feature[v].size(1)).to(device)  # 创建零张量
            #         low_feature[v] = torch.cat((low_feature[v], padding_tensor), dim=0)  #
            print("计算准确率的low_feature的shape",low_feature[v].shape)
            common_feature=model.forward_common_future(low_feature,p_sample,adaptive_weight)
            common_feature = common_feature.detach()
            cluster_common_feature.extend(common_feature.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    cluster_common_feature = np.array(cluster_common_feature)
    return labels_vector, cluster_common_feature

def valid(model, device, dataset, view, data_size, class_num, p_sample,adaptive_weight):
    #测试集
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    # # 获取真实标签和用于聚类的矩阵
    # labels_vector, common_feature = inference(test_loader, model, device, view, data_size, p_sample,adaptive_weight)

    # # #新增
    labels_vector, common_feature = inference_tsne(test_loader, model, device, view, data_size, p_sample,adaptive_weight)
    print('---------train over---------')
    print('Clustering results:')
    #利用kemeans聚类
    print("最后用于聚类的common_feature的大小",common_feature.shape)
    print("最后用于聚类的labels_vector的大小",labels_vector.shape)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(common_feature)
    print("最后用于聚类的y_pred的大小",y_pred.shape)
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))

def new_valid(model, device, dataset, view, data_size, class_num, p_sample,adaptive_weight,tot_loss):
    #测试集
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    #获取真实标签和用于聚类的矩阵
    labels_vector, common_feature = inference(test_loader, model, device, view, data_size, p_sample,adaptive_weight)
    print('---------train over---------')
    print('Clustering results:')
    #利用kemeans聚类
    print("最后用于聚类的common_feature的大小",common_feature.shape)
    print("最后用于聚类的labels_vector的大小",labels_vector.shape)
    kmeans = KMeans(n_clusters=class_num, n_init=100)
    y_pred = kmeans.fit_predict(common_feature)
    print("最后用于聚类的y_pred的大小",y_pred.shape)
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))

    results = [acc, nmi, pur, ari, tot_loss]
 

    # Return all_results if needed
    return results
