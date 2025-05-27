import torch
from network import Network,MultiHeadAttention, FeedForwardNetwork
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os
import torch.nn.functional as F


Dataname = 'Hdigit'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=1.0)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0001)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=0)
parser.add_argument("--con_epochs", default=5)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--extract_feature_dim", default=512)
parser.add_argument("--seed", type=int, default=15)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--ffn_size', type=int, default=32)
parser.add_argument('--attn_bias_dim', type=int, default=6)
parser.add_argument('--attention_dropout_rate', type=float, default=0.5)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if args.dataset == "MNIST-USPS":
#     args.con_epochs = 100
#     seed = 10
# if args.dataset == "CCV":
#     args.con_epochs = 100
#     seed = 3
# if args.dataset == "Hdigit":
#     args.con_epochs =100
#     seed = 10
# if args.dataset == "YouTubeFace":
#     args.con_epochs = 100
#     seed = 10
# if args.dataset == "Cifar10":
#     args.con_epochs = 10
#     seed = 10
# if args.dataset == "Cifar100":
#     args.con_epochs = 200
#     seed = 10
# if args.dataset == "Prokaryotic":
#     args.con_epochs = 50
#     seed = 10
# if args.dataset == "Synthetic3d":
#     args.con_epochs = 100
#     seed = 10
# if args.dataset == "Caltech-2V":
#     args.con_epochs = 100
#     seed = 10
# if args.dataset == "Caltech-3V":
#     args.con_epochs = 100
#     seed = 10
# if args.dataset == "Caltech-4V":
#     args.con_epochs = 150
#     seed = 10
# if args.dataset == "Caltech-5V":
#     args.con_epochs = 200
#     seed = 5
# 以Caltech-2V为例子
# dataset
# dim
# view 
# data_size
# class_num
dataset, dims, view, data_size, class_num = load_data(args.dataset)

# 在这里，seed是用于生成随机种子的数值。
# torch.manual_seed(seed)用于设置PyTorch的随机种子，torch.cuda.manual_seed_all(seed)
# 用于设置所有CUDA设备的随机种子，np.random.seed(seed)用于设置NumPy的随机种子，
# random.seed(seed)用于设置Python标准库中random模块的随机种子，
# torch.backends.cudnn.deterministic = True用于确保使用的cuDNN可确定性，这样在相同条件下，每次运行的结果都是一样的。
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

#预训练函数
def pretrain(epoch):
    tot_loss=0.
    for batch_idx,(x,_,_)in enumerate(data_loader):
        for v in range(view):
            x[v]=x[v].to(device)
        optimizer.zero_grad()
        _,_,decoder_feature=model(x)
        loss_list=[]
        for v in range(view):
            loss_list.append(F.mse_loss(x[v],decoder_feature[v]))
        loss=sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss+=loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def train(epoch, p_sample, adaptive_weight):
    tot_loss=0.
    for batch_idx,(x,_,_) in enumerate(data_loader):
        for v in range(view):
            x[v]=x[v].to(device)
        optimizer.zero_grad()

        #获取私有信息,低级潜在表示，解码信息
        private_x,low_feature, decoder_feature=model.forward(x)
        loss_list = []
        
        #获取公共信息
        common_feature=model.forward_common_future(low_feature,p_sample,adaptive_weight)


        #计算重建损失和私有信息之间的损失
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(criterion.forward_private(private_x[v], private_x[w]))
            loss_list.append(F.mse_loss(x[v],decoder_feature[v]))
        #计算公有信息与私有信息之间的损失 
        # for v in range(view):
        #     loss_list.append(criterion.forward_common(common_feature,private_x[v] ))

        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()

    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


#搭建模型
attention_net = MultiHeadAttention(args.hidden_dim, args.attention_dropout_rate,args.num_heads, args.attn_bias_dim)
p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate)
model = Network(view, dims, args.feature_dim, args.extract_feature_dim, class_num, device, attention_net, p_net)

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# #暂时使用注意力机制模块
# attention_net = MultiHeadAttention(args.hidden_dim, args.attention_dropout_rate, args.num_heads, args.attn_bias_dim)

# attention_net = attention_net.to(device)

# optimizer_atten_net = torch.optim.Adam(attention_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

# #前馈神经网络，用于计算权重
# p_net = FeedForwardNetwork(view, args.ffn_size, args.attention_dropout_rate)

# p_net = p_net.to(device)

# optimizer_p_net = torch.optim.Adam(p_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

#loss函数（暂时使用MFLVC）
criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)


# 视图采样概率
p_sample = np.ones(view)
weight_history = []
p_sample = p_sample / sum(p_sample)
p_sample = torch.FloatTensor(p_sample).cuda()


# 初始化自适应权重
adaptive_weight = np.ones(view)
adaptive_weight = adaptive_weight / sum(adaptive_weight)
adaptive_weight = torch.FloatTensor(adaptive_weight).cuda()
adaptive_weight = adaptive_weight.unsqueeze(1)


# 开始训练
epoch = 1
#预训练阶段
while epoch <= args.mse_epochs:
    pretrain(epoch)
    epoch += 1
#训练阶段
while epoch <= args.mse_epochs + args.con_epochs:
    train(epoch, p_sample,adaptive_weight)
    #验证阶段
    if epoch == args.mse_epochs + args.con_epochs:
        valid(model, device, dataset, view, data_size, class_num, p_sample,adaptive_weight)
    epoch += 1