import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,input_dim,feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)



class Network(nn.Module):
    def __init__(self,view,input_size,feature_dim,extract_feature_dim,class_num,device,MultiHeadAttention,FeedForwardNetwork):
        super(Network,self).__init__()
        self.encoders=[]
        self.decoders=[]
        for v in range(view):
            self.encoders.append(Encoder(input_size[v],feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v],feature_dim).to(device))
        self.encoders=nn.ModuleList(self.encoders)
        self.decoders=nn.ModuleList(self.decoders)

        self.extract_private=nn.Sequential(
            nn.Linear(feature_dim,extract_feature_dim),
        )

        self.attention_net=MultiHeadAttention
        self.p_net=FeedForwardNetwork

        self.view=view

        self.extract_common=nn.Sequential(
            nn.Linear(feature_dim,extract_feature_dim),
        )

        self.Common_view = nn.Sequential(
            nn.Linear(feature_dim, extract_feature_dim),
        )
        
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
    
    def forward(self,x):
        # 私有信息
        private_x=[]
        # 低级特征
        low_feature=[]
        decoder_feature=[]

        for v in range(self.view):
            #对每个视图进行处理
            x_view = x[v]
            z = self.encoders[v](x_view)
            #获取私有信息
            private_z = normalize(self.extract_private(z), dim=1)
            #获取decoder后的重建信息
            zd = self.decoders[v](z)

            decoder_feature.append(zd)
            private_x.append(private_z)
            low_feature.append(z)

        #GCFAgg的简单拼接获取公共信息，但这里先不返回，因为训练所使用的是dealMVC
        # commonz = torch.cat(low_feature, 1)
        # commonz, S = self.TransformerEncoderLayer(commonz)
        # commonz = normalize(self.Common_view(commonz), dim=1)
        return private_x, low_feature, decoder_feature

    def forward_common_future(self,low_feature,p_sample,adaptive_weight):
        #获取公有信息(这里使用DealMVC的融合方式，但是使用的是低级潜在表示)
        #DealMVC是通过获取后的私有信息融合公共信息
        #1.使用GCFACG的融合方式（未尝试）
        #2.使用DealMVC的融合方式，但是使用的是低级潜在表示（已尝试）
        #3.简单拼接
        
        
        common_x=[]
        # for v in range(self.view):
        #     demo=self.TransformerEncoder(low_feature[v])
        #     demo=normalize(demo, dim=1)
        #     demo=self.Common_view(demo)
        #     demo=normalize(demo, dim=1)
        #     common_x.append(normalize(demo, dim=1))

        for v in range(self.view):
            demo=self.TransformerEncoder(low_feature[v])
            common_x.append(normalize(demo, dim=1))

            

        if(low_feature[0].shape[0]==256):
            print("使用了注意力机制",low_feature[0].shape[0])
            zs_tensor = torch.tensor([]).cuda()

            # 对于每个视图（v）
            for v in range(self.view):
                # 计算每个视图的低级潜在表示的平均值，并调整维度以获得形状（d * v）
                zs_tensor = torch.cat((zs_tensor, torch.mean(common_x[v], 1).unsqueeze(1)), 1)
            # 转置后维度是（2，batch_size）
            # 转置张量
            zs_tensor = zs_tensor.t()

            # 首先使用注意力网络计算注意力权重（v * 1）
            zs_tensor = self.attention_net(zs_tensor, zs_tensor, zs_tensor)
            # 使用前馈神经网络p_net计算p_sample的学习概率（v * 1）
            p_learn = self.p_net(p_sample)


            # 计算r作为注意力和学习概率的乘积
            r = zs_tensor * p_learn
            # 使用softmax函数归一化r
            s_p = nn.Softmax(dim=0)
            r = s_p(r)


            # 将自适应权重与注意力权重相乘
            adaptive_weight = r * adaptive_weight


            #公共表示
            common_feature = torch.zeros([common_x[0].shape[0], common_x[0].shape[1]]).cuda()
            # 对于每个视图（v）
            for v in range(self.view):
                # 计算融合特征，乘以对应的自适应权重并累加
                common_feature = common_feature + adaptive_weight[v].item() * common_x[v]
        else:
            #公共表示
            common_feature = torch.zeros([common_x[0].shape[0], common_x[0].shape[1]]).cuda()
            # 对于每个视图（v）
            for v in range(self.view):
                # 计算融合特征，乘以对应的自适应权重并累加
                common_feature = common_feature + adaptive_weight[v].item() * common_x[v]


    #    if(low_feature[0].shape[0]==256):
    #         print("使用了注意力机制",low_feature[0].shape[0]) 
    #         zs_tensor = torch.tensor([]).cuda()

    #         # 对于每个视图（v）
    #         for v in range(self.view):
    #             # 计算每个视图的低级潜在表示的平均值，并调整维度以获得形状（d * v）
    #             zs_tensor = torch.cat((zs_tensor, torch.mean(low_feature[v], 1).unsqueeze(1)), 1)
    #         # 转置后维度是（2，batch_size）
    #         # 转置张量
    #         zs_tensor = zs_tensor.t()
    #         # 首先使用注意力网络计算注意力权重（v * 1）
    #         zs_tensor = self.attention_net(zs_tensor, zs_tensor, zs_tensor)
    #         # 使用前馈神经网络p_net计算p_sample的学习概率（v * 1）
    #         p_learn = self.p_net(p_sample)


    #         # 计算r作为注意力和学习概率的乘积
    #         r = zs_tensor * p_learn
    #         # 使用softmax函数归一化r
    #         s_p = nn.Softmax(dim=0)
    #         r = s_p(r)

    #         # 将自适应权重与注意力权重相乘
    #         adaptive_weight = r * adaptive_weight

    #         #公共表示
    #         common_feature = torch.zeros([low_feature[0].shape[0], low_feature[0].shape[1]]).cuda()
    #         # 对于每个视图（v）
    #         for v in range(self.view):
    #             # 计算融合特征，乘以对应的自适应权重并累加
    #             common_feature = common_feature + adaptive_weight[v].item() * low_feature[v]
    #     else:
    #         #公共表示
    #         common_feature = torch.zeros([low_feature[0].shape[0], low_feature[0].shape[1]]).cuda()
    #         # 对于每个视图（v）
    #         for v in range(self.view):
    #             # 计算融合特征，乘以对应的自适应权重并累加
    #             common_feature = common_feature + adaptive_weight[v].item() * low_feature[v]


        # common_feature = self.TransformerEncoderLayer(common_feature)

        # common_feature = normalize(self.Common_view(common_feature), dim=1)
       
        return common_feature

class FeedForwardNetwork(nn.Module):
    def __init__(self, view, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(view, ffn_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_size, view)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = x.unsqueeze(1)
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, 1)
    #实际传进来的是q,k,v一样（2，256）
    def forward(self, q, k, v):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]


        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)

        x = self.output_layer(x)

        return x


    

