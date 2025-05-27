import torch
import torch.nn as nn
import math
import sys
class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0

        # 针对四维特征张量，需要在两个维度上进行修改
        for i in range(N):
            mask[i, i] = 0

        mask = mask.bool()
        return mask


    def forward_private(self, h_i, h_j):
        N = 2 * self.batch_size         # 表示总共的样本数，即正负样本的总和
        h = torch.cat((h_i, h_j), dim=0)    # 是将两个输入特征向量连接在一起，以便后续计算相似度。

        sim = torch.matmul(h, h.T) / self.temperature_f   # 是一个相似度矩阵，通过将特征向量进行内积并除以一个温度参数来计算。

        # 是从相似度矩阵 sim 中提取出来的对角线元素，分别表示样本 i 和样本 j 之间的相似度。
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        #是由 sim_i_j 和 sim_j_i 构成的正样本对的相似度向量，被拉平成一个列向量
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        #是一个掩码，用于选择 sim 中的一部分元素，这些元素将被用作负样本对的相似度。
        mask = self.mask_correlated_samples(N)

        # 是从 sim 中根据 mask 选取的负样本对的相似度，同样被拉平成一个列向量。
        negative_samples = sim[mask].reshape(N, -1)

        #labels 是一个全零的标签向量，长度为 N，用于指示正样本和负样本
        labels = torch.zeros(N).to(positive_samples.device).long()
        #logits 是将正样本和负样本的相似度拼接在一起形成的向量。
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        #loss 是通过将 logits 和 labels 输入交叉熵损失函数 self.criterion 计算得到的损失。
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_common(self, h_i, h_j):
        N = 2 * self.batch_size         # 表示总共的样本数，即正负样本的总和
        h = torch.cat((h_i, h_j), dim=0)    # 是将两个输入特征向量连接在一起，以便后续计算相似度。

        sim = torch.matmul(h, h.T) / self.temperature_l   # 是一个相似度矩阵，通过将特征向量进行内积并除以一个温度参数来计算。

        # 是从相似度矩阵 sim 中提取出来的对角线元素，分别表示样本 i 和样本 j 之间的相似度。
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        #是由 sim_i_j 和 sim_j_i 构成的正样本对的相似度向量，被拉平成一个列向量
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        #是一个掩码，用于选择 sim 中的一部分元素，这些元素将被用作负样本对的相似度。
        mask = self.mask_correlated_samples(N)

        # 是从 sim 中根据 mask 选取的负样本对的相似度，同样被拉平成一个列向量。
        negative_samples = sim[mask].reshape(N, -1)

        #labels 是一个全零的标签向量，长度为 N，用于指示正样本和负样本
        labels = torch.zeros(N).to(positive_samples.device).long()
        #logits 是将正样本和负样本的相似度拼接在一起形成的向量。
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        #loss 是通过将 logits 和 labels 输入交叉熵损失函数 self.criterion 计算得到的损失。
        loss = self.criterion(logits, labels)
        loss /= N
        return loss