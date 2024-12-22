#encoding=utf-8
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils import weight_norm
import time
from model.layers import TemporalConvNet, SpatialSelfAttention, TemporalSelfAttention, STSelfAttention, MulSelfAttention, Block

def SAM(x, y):

    x_shapes = np.array([x[i:i+3] for i in range(len(x)-2)])
    y_shapes = np.array([y[i:i+3] for i in range(len(y)-2)])

    # 
    distances = np.zeros((len(x_shapes), len(y_shapes)))
    for i in range(len(x_shapes)):
        for j in range(len(y_shapes)):
            distances[i, j] = np.linalg.norm(x_shapes[i]-y_shapes[j])

    average_distance = np.mean(distances)

    #
    normalized_distance = average_distance / (np.linalg.norm(x[-3:]) + np.linalg.norm(y[-3:]))

    # 
    return normalized_distance

def euc_dis(x, y):  
    return np.sqrt(np.sum((x - y) ** 2)) 

def cos_sim(vec1, vec2):  
    dot_product = np.dot(vec1, vec2)  
    norm_vec1 = np.linalg.norm(vec1)  
    norm_vec2 = np.linalg.norm(vec2)  
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)  
    return cosine_similarity  


def get_DynGraph(data):
    
    B, N, _ = data.shape
    #
    fun_graph =  np.zeros((N, N))
    #batch_size
    his_win = np.zeros((N, 11 + B))
    
    flag = 0
    for b in range(B):
        if b == 0:
            flag = 11
            his_win[:,0:12] = data[0].cpu().numpy()
        else:
            flag = flag + 1 
            his_win[:, flag] = data[b,:,-1].cpu().numpy()

    for n in range(N):
        
        for m in range(N):
            tmp_dis = euc_dis(his_win[n], his_win[m])
            #tmp_dis = cos_sim(his_win[n], his_win[m])
            
            fun_graph[n][m] = tmp_dis
            fun_graph[m][n] = tmp_dis
    
    # print("F",np.isnan(fun_graph).any())
    sem_mask = fun_graph.argsort(axis=1)[:, -10:]

    adj = np.zeros((N, N))

    for i in range(n):
        adj[i][sem_mask[i]] = 1
    
    #print('adj',np.isnan(adj).any())
    return torch.tensor(fun_graph, dtype=torch.float32).to(data.device)



def positional_encoding(X, num_features, dropout_p=0.1, max_len=12):
    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1,max_len,num_features))
    X_ = torch.arange(max_len,dtype=torch.float32).reshape(-1,1) / torch.pow(
        10000,
        torch.arange(0,num_features,2,dtype=torch.float32) /num_features)
    P[:,:,0::2] = torch.sin(X_)
    P[:,:,1::2] = torch.cos(X_)
    return P.to(X.device)
         

#embding
class TimePeriodEmb(nn.Module):
    def __init__(self, devices, emb_dim):
        super(TimePeriodEmb, self).__init__()

        self.minute_size = 289
        self.weekday_size = 8
        
        self.emb_dim = emb_dim
        self.device = devices
        self.daytime_embedding = nn.Embedding(self.minute_size, self.emb_dim).to(devices)
        self.weekday_embedding = nn.Embedding(self.weekday_size, self.emb_dim).to(devices)
        

    def forward(self, x_day, x_week):
    
        # 
        out_day = self.daytime_embedding(x_day.long().to(self.device))
        # 
        out_week =  self.weekday_embedding(x_week.long().to(self.device))
        
        return out_day + out_week


def get_laplacian_vex(graph):
    
    print("G",torch.isnan(graph).any())

    D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
    print("D",torch.isnan(D).any())
    
    M = torch.mm(torch.mm(D, graph), D)
    print("M",torch.isnan(M).any())

    L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - M
    print("L", torch.isnan(L).any())

    _, eig_vex = np.linalg.eig(L.cpu())
    #_, eig_vex = torch.linalg.eig(L)

    return torch.tensor(eig_vex, dtype = torch.float)

class TCNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes, kernel_size=3, dropout=0.1):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, hidden_sizes[:3], kernel_size, dropout)
        self.act = nn.ReLU()
        self.linear = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        output_time_step = 1
        y_t = self.tcn.network(x)
        y_t = self.act(y_t)
        y_t = self.linear(y_t)
        # y = y_t[:, :, -output_time_step]
        return y_t



class DPSTCN(nn.Module):
    def __init__(self, t_inputs, t_channels, graph, device, n_vertex=307, kernel_size_tcn=2, dropout=0.3, g_inputs=1,  hidden=64, g_output=32,  output_size=12, layers=1):
        super(DPSTCN,self).__init__()
        
        # 
        self.device = device
        self.n_vertex = n_vertex
        self.graph = graph

        self.layers = layers

        # 
        self.adj_linear = nn.Linear(n_vertex, 16)
        self.timeEmb = TimePeriodEmb(devices=device, emb_dim=16).to(device)
        
        # 
        self.mulatt = MulSelfAttention(8, 16, 16, device)
        
        self.Smulatt = SpatialSelfAttention(8, 16, 16, device)

        # 
        self.gtconv = Block(t_inputs, t_channels, device, kernel_size_tcn, dropout, n_vertex, g_inputs, hidden, g_output, k = 2).to(device)
        
        self.datadrop = nn.Dropout(dropout)
        
        # 
        self.outlayer= nn.ModuleList()
        for k in range(n_vertex):
            self.outlayer.append(nn.Sequential(
                nn.Linear(32, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ))
        
    def forward(self, x):
        
        x_ori = x['flow_x']


        B, N, _ = x_ori.shape
        
        fun_graph = get_DynGraph(x_ori).to(self.device)
        
        graphs = [fun_graph, self.graph]

            
        x_pe = positional_encoding(x_ori, 16)
        x_te = self.timeEmb(x['day_cyc'], x['week_cyc']).unsqueeze(1)
        
        x_input_t = x_ori.unsqueeze(-1) + x_pe + x_te
        
        #x_lapE = self.adj_linear(get_laplacian_vex(fun_graph).to(self.device)).unsqueeze(1)
        
        x_input_g = x_ori.unsqueeze(-1)
            

        x_tcn = x_input_t + self.mulatt(x_input_t)
        
        x_gcn = x_input_g
        
        hid_fea = self.gtconv(x_gcn, x_tcn, graphs)


        B, N, L, C = hid_fea.shape
        
        
        out = torch.zeros([B, N, 12],dtype=torch.double).to(self.device)

        for i in range(self.n_vertex):
        
            out[:,i,:] = self.outlayer[i](hid_fea[:,i,:,:].float()).squeeze()


        return out
