#encoding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import math
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
   
        return x[:, :, :-self.chomp_size, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):

        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 
        self.chomp1 = Chomp1d(padding)  # 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):

        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.0):

        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  
            out_channels = num_channels[i]  
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):

        x = x.permute(0, 3, 1, 2)
        return self.network(x)

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        """
        GCN
        :param in_c: input channels
        :param hid_c:  hidden nodes
        :param out_c:  output channels
        """
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data):
        graph_data = data["graph"][0]  # [N, N]
        graph_data = self.process_graph(graph_data)
        flow_x = data["flow_x"]  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]
        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)


class ChebConv(nn.Module):

    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        """
        ChebNet conv
        :param in_c: input channels
        :param out_c:  output channels
        :param K: the order of Chebyshev Polynomial
        :param bias:  if use bias
        :param normalize:  if use norm
        """
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """

        :param inputs: he input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)  # [K, 1, N, N]
        
        
        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        #print(self.weight.shape)
    
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian

        :param laplacian: the multi order Chebyshev laplacian, [K, N, N]
        :return:
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)
        
        largest_eigval, _ = torch.linalg.eigh(laplacian)
        largest_eigval = torch.max(largest_eigval)

        scaled_laplacian = (2. / largest_eigval) * laplacian - torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = scaled_laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    k1 = k - 1
                    k2 = k - 2
                    multi_order_laplacian[k] = 2.0 * torch.mm(scaled_laplacian.clone().float(), multi_order_laplacian[k1].clone()) - multi_order_laplacian[k2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        compute the laplacian of the graph
        :param graph: the graph structure without self loop, [N, N]
        :param normalize: whether to used the normalized laplacian
        :return:
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L


class ChebNet(nn.Module):

    def __init__(self, in_c, hid_c, out_c, K, n_vertex=170, dropout=0.0):
        """
        :param in_c: int, number of input channels.
        :param hid_c: int, number of hidden channels.
        :param out_c: int, number of output channels.
        :param K:
        """
        super(ChebNet, self).__init__()
    
        self.drop = nn.Dropout(dropout)
        
        self.conv1 = ChebConv(in_c=in_c, out_c=out_c, K=K)
        
        self.act = nn.ReLU()
    

    def forward(self, data, graph):
        #graph_data = data["graph"][0]  # [N, N]
        flow_x = data  # [B, N, H, C]

        B, N, T, C= flow_x.size()

        flow_x = flow_x.reshape(B, N, -1)  # [B, N, H*D]
        
        out = torch.zeros([B, N, 12, 32],dtype=torch.float64).to(data.device)
        for i in range(12):
            out[:,:,i,:] = self.act(self.conv1(data[:,:,i,:], graph))
        

        out = self.drop(out)


        return out

class MulSelfAttention(nn.Module):
    def __init__(self,  num_heads, key_dim, d_model, device):
        super(MulSelfAttention, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device

 
        #self.te = TimePeriodEmb(device, 1)
        self.peL = nn.Linear(key_dim * key_dim, key_dim).to(device)
        self.teL = nn.Linear(key_dim * key_dim, key_dim).to(device)

        self.wq = []
        self.wk = []
        self.wv = []

        for i in range(self.num_heads):
            self.wq.append(nn.Conv2d(key_dim, d_model, 7, stride=1, padding=3).to(device))
            self.wk.append(nn.Conv2d(key_dim, d_model, 7, stride=1, padding=3).to(device))
            self.wv.append(nn.Conv2d(key_dim, d_model, 1, stride=1).to(device))


        self.ll = nn.Linear(self.num_heads * d_model, d_model).to(device)

    def forward(self, x, mask = None):
        
        B, N, L, _ = x.shape

        mul_att = torch.zeros([B, N, L, self.num_heads, self.d_model], dtype=torch.float).to(self.device)

        for n in range(self.num_heads):
            # Q、K、V
            Q = self.wq[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            K = self.wk[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            V = self.wv[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        
            att = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_model)

            att = torch.softmax(att, dim=-1)

            mul_att[:, :, :, n, :] = torch.matmul(att, V)


        output = self.ll(mul_att.reshape(B, N, L, self.num_heads * self.d_model))
        
        return output

class SpatialSelfAttention(nn.Module):
    def __init__(self,  s_num_heads, in_c, dim, device, qkv_bias=False,attn_drop=0., proj_drop=0.):

        super().__init__()
        self.num_heads = s_num_heads
        self.d_model = dim
        self.k_dim = in_c
        self.device = device

        self.wq = []
        self.wk = []
        self.wv = []

        for i in range(self.num_heads):
            self.wq.append(nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=qkv_bias).to(device))
            self.wk.append(nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, bias=qkv_bias).to(device))
            self.wv.append(nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=qkv_bias).to(device))
        
        self.s_attn_drop = nn.Dropout(attn_drop)
        
        self.ll = nn.Linear(self.num_heads * dim, dim).to(device)

    def forward(self, x, mask=None):

        B, T, N, D = x.shape
        
        mul_att = torch.zeros([B, T, N, self.num_heads, self.d_model], dtype=torch.float).to(self.device)

        for n in range(self.num_heads):
            
            Q = self.wq[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            K = self.wk[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            V = self.wv[n](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

            
            att = torch.matmul(Q, K.permute(0, 1, 3, 2)) / math.sqrt(self.d_model)

            att = torch.softmax(att, dim=-1)
            
            mul_att[:, :, :, n, :] = torch.matmul(att, V)


        output = self.ll(mul_att.reshape(B, N, T, self.num_heads * self.d_model))


        return output

class TemporalSelfAttention(nn.Module):
    def __init__(
        self, dim, ratio, t_num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0. ):
        super().__init__()
        assert dim % t_num_heads == 0
        self.t_num_heads = t_num_heads
        self.head_dim = dim // t_num_heads
        self.scale = self.head_dim ** -0.5
        self.ratio = ratio

        self.t_q_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)


    def forward(self, x, mask=None):

        B, T, N, D = x.permute(0, 2, 1, 3).shape
        t_q = self.t_q_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_k = self.t_k_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_v = self.t_v_conv(x.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)
        t_q = t_q.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_k = t_k.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        t_v = t_v.reshape(B, N, T, self.t_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        t_attn = (t_q @ t_k.transpose(-2, -1)) * self.scale
    
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)

        t_x = (t_attn @ t_v).transpose(2, 3).reshape(B, N, T, D).transpose(1, 2)

        return t_x

class STSelfAttention(nn.Module):
    def __init__(self, dim, dim_out, t_num_heads=4, s_num_heads=4, proj_drop=0.1):
        super().__init__()
        self.tratio = t_num_heads / (t_num_heads + s_num_heads)
        self.sratio = s_num_heads / (t_num_heads + s_num_heads)
        self.dim = dim
        self.dim_out = dim_out
        self.t_num_heads = t_num_heads
        self.s_num_heads = s_num_heads
        
        self.tatt = TemporalSelfAttention(dim=self.dim, ratio=self.tratio, t_num_heads=self.t_num_heads)
        self.satt = SpatialSelfAttention(dim=self.dim, ratio=self.sratio, s_num_heads=self.s_num_heads)
        
        self.gate = nn.Sequential(
            nn.Linear(dim*2, dim_out),
            nn.Sigmoid()
        )

        #self.proj = nn.Linear(dim*2, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, T, D = x.size()
        t_x = self.tatt(x).permute(0, 2, 1, 3)
        s_x = self.satt(x)
          
        fus_att = torch.cat([t_x, s_x], dim=-1)
        
        x = self.gate(fus_att) * t_x + (1 - self.gate(fus_att) * s_x)
        x = self.proj_drop(x)
        
        return x

class Block(nn.Module):

    def __init__(self, t_inputs, t_channels, device, kernel_size_tcn, dropout, n_vertex, g_inputs, hidden, g_output, k, d=32):
        super(Block, self).__init__()

        self.tcn = TemporalConvNet(t_inputs, t_channels, kernel_size_tcn, dropout)
        #self.tcn1 = TemporalConvNet(t_inputs, t_channels, kernel_size_tcn, dropout)
        self.relu = nn.ReLU()
        self.d_model = d
        self.graph_conv = ChebNet(g_inputs, hidden, g_output, k, dropout)
        self.fung_conv = ChebNet( g_inputs, hidden, g_output, k, dropout)


        self.gate = nn.Sequential(
            nn.Linear(d*2, d),
            nn.Sigmoid()
        )

        

        self.outdrop = nn.Dropout(dropout)
        
    def forward(self, x_gcn, x_tcn, graphs):
        
        B, N, L, _ = x_gcn.shape
        
        
        out_tcn = self.tcn(x_tcn).permute(0, 2, 3, 1)

        out_fgcn = self.fung_conv(x_gcn, graphs[0]).reshape(B, N, L, self.d_model)
        out_ggcn = self.graph_conv(x_gcn, graphs[1]).reshape(B, N, L, self.d_model)
        
        out_gcn = out_fgcn + out_ggcn
        #out_gcn = out_fgcn

        fus = torch.cat((out_gcn, out_tcn), dim=-1).float()
        out = self.gate(fus) * out_tcn + (1 - self.gate(fus)) * out_gcn
        #out = out_tcn + out_gcn
        out = self.relu(out)
        
        return out
# encoding=utf-8
