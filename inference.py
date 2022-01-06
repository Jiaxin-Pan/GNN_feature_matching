import os
import open3d as o3d
import numpy as np
import torch
from torch import tensor
import math
import random
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_cluster import fps
from torch_geometric.transforms import Compose, LinearTransformation, RandomRotate
from torch_cluster import knn_graph
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import EdgeConv, PPFConv
from torch.nn import Sequential, Linear, ReLU

def get_ratio(num_data):
    if num_data > 1000:
        ratio = 0.05
    elif 500 < num_data < 1000:
        ratio = 0.1
    elif 100 < num_data < 500:
        ratio = 0.5
    else:
        ratio = 1
    return ratio

def random_rotation(data):
    torch.manual_seed(123)
    random_rotate = Compose([
        RandomRotate(degrees=180, axis=0),
        RandomRotate(degrees=180, axis=1),
        RandomRotate(degrees=180, axis=2),
    ])
    return random_rotate(data)

def cosine_distance(input1, input2):
    norm1 = F.normalize(input1, 2, 1)
    norm2 = F.normalize(input2, 2, 1)
    cosine_similarity = torch.matmul(norm1,norm2.T)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def contrastive_loss(origin_output, transform_output):
    num_nodes = origin_output.shape[0]
    dim_features = origin_output.shape[1]
    #compute the distance matrix
    pmatrix = cosine_distance(origin_output, transform_output)
    #compute the contrastive loss
    diagonal = pmatrix * torch.eye(num_nodes)
    pos_dis = torch.sum(torch.square(diagonal), dim = (0,1)) / num_nodes

    non_diagonal = pmatrix - diagonal
    margin_matrix = margin * (torch.ones(num_nodes) - torch.eye(num_nodes))
    non_diagonal_margin =  margin_matrix - non_diagonal
    non_diagonal_dis = torch.maximum(torch.zeros(num_nodes), non_diagonal_margin)
    neg_dis = torch.sum(torch.square(non_diagonal_dis), dim = (0,1)) / (num_nodes * (num_nodes-1))
    return pos_dis + neg_dis, pmatrix

def load_GraphUNET():
    GNNmodel = GraphUNet(in_channels = 3, hidden_channels = 128, out_channels = 512, depth = 4, \
                  pool_ratios=0.5, sum_res=True, act = torch.nn.functional.relu)
    PATH = "./models/GNNmodel.p"
    GNNmodel_load = GraphUNet(in_channels = 3, hidden_channels = 128, out_channels = 512, depth = 4, \
                          pool_ratios=0.5, sum_res=True, act = torch.nn.functional.relu)
    GNNmodel_load.load_state_dict(torch.load(PATH))
    GNNmodel_load.eval()
    return GNNmodel_load

class EdgeGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeGCN, self).__init__()
        torch.manual_seed(1234567)
        self.sequential = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.conv1 = EdgeConv(self.sequential)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x    
    
def load_EdgeModel():
    PATH = "./models/Edgemodel.p"
    EdgeModel_load = EdgeGCN(in_channels= 3*2, out_channels=512)
    EdgeModel_load.load_state_dict(torch.load(PATH))
    EdgeModel_load.eval()
    return EdgeModel_load

class PPFGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PPFGCN, self).__init__()
        torch.manual_seed(1234567)
        self.local_sequential = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.global_sequential = Sequential(
            Linear(out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels)
        )
        self.conv1 = PPFConv(local_nn = self.local_sequential, global_nn = self.global_sequential)

    def forward(self, x, edge_index):
        x = self.conv1(x = x, pos = x, normal = x, edge_index = edge_index)
        return x

def load_PPFModel():
    PATH = "./models/PPFmodel.p"
    PPFModel_load = PPFGCN(in_channels= 3+4, out_channels=512)
    PPFModel_load.load_state_dict(torch.load(PATH))
    PPFModel_load.eval()
    return PPFModel_load

def acc_distance(origin_output, transform_output, num_points):
    pmatrix =  cosine_distance(origin_output, transform_output)
    match = torch.argmin(pmatrix, 0)
    num_match = 0
    for i in range(len(match)):
        if match[i] == i:
            num_match += 1
    test_acc = num_match / num_points
    test_acc = test_acc *100
    print("Argmin accuracy: {}%".format(test_acc))
    return test_acc

def acc_Hungarian(origin_output, transform_output, num_points):
    pmatrix =  cosine_distance(origin_output, transform_output)
    row_idx, col_idx = linear_assignment(pmatrix.detach().numpy())
    num_match = torch.sum(torch.where(torch.tensor(row_idx) == torch.tensor(col_idx),1,0))
    test_acc = num_match / num_points
    test_acc = test_acc.item()*100
    print("Hungarian accuracy: {}%".format('%.1f' % test_acc))
    return test_acc, row_idx, col_idx

def random_test(data, row_idx, col_idx):
    num_points = data.pos.shape[0]
    picked_idx = np.random.choice(len(data.pos), 6, replace=False)
    picked_row = row_idx[picked_idx]
    picked_col = col_idx[picked_idx]
    print("piecked_row", picked_row)
    print("picked_col",picked_col)
    num_match = torch.sum(torch.where(torch.tensor(picked_row) == torch.tensor(picked_col),1,0))
    test_acc = num_match / 6
    test_acc = test_acc.item()*100
    print("{}% of randomly chosen points are matched correctly".format('%.1f' % test_acc))
    #change the color for the chosen points
    pos_color = torch.zeros(num_points, 3)
    postr_color = torch.zeros(num_points, 3)
    for i in range(len(picked_idx)):
        color_idx = i %3
        pos_color[picked_row[i]][color_idx] = int(i/3) * 0.3 + 0.7
        postr_color[picked_col[i]][color_idx] = int(i/3) * 0.3 + 0.7
    #get the 3d point clouds    
    point_cloud1 = o3d.geometry.PointCloud()
    point_cloud1.points = o3d.utility.Vector3dVector(data.pos)
    point_cloud1.colors = o3d.utility.Vector3dVector(pos_color)
    point_cloud2 = o3d.geometry.PointCloud()
    point_cloud2.points = o3d.utility.Vector3dVector(data.pos_test_tr)
    point_cloud2.colors = o3d.utility.Vector3dVector(postr_color)
    return point_cloud1, point_cloud2
    
class GetMatching():
    def __init__(self, ):
        super().__init__()
  
    def __call__(self, data, which_model = 'PPFModel'):
        
        if which_model == 'GraphUNET':
            model = load_GraphUNET()
        elif which_model == 'EdgeModel':
            model = load_EdgeModel()
        elif which_model == 'PPFModel':
            model = load_PPFModel()
        
        #downsample the original data
        num_data = data.pos.shape[0]  
        ratio = get_ratio(num_data)
        index = fps(data.pos, ratio=ratio)  
        data.pos = data.pos[index]
        num_points = data.pos.shape[0]

        #create the edges
        data.edge_index = knn_graph(data.pos, k=6)

        #augment the point cloud randomly
        #random rotation
        data_tr = data.clone().detach()
        data_tr = random_rotation(data_tr)

        #random translation
        random_translation = torch.rand(1, 3) #*0.1
        data_tr.pos = data_tr.pos + random_translation
        data.pos_test_tr = data_tr.pos.clone().detach()

        # get the embeddings from the model
        origin_output = model(data.pos, data.edge_index)
        transform_output = model(data.pos_test_tr, data.edge_index)
        
        #compute the matching points and the accuracy with argmin Matching
        origin_acc = acc_distance(origin_output, transform_output, num_points)

        #compute the matching points and the accuracy with Hungarian Matching
        Hungarian_acc, row_idx, col_idx = acc_Hungarian(origin_output, transform_output, num_points)

        #pick 6 points randomly for test
        point_cloud1, point_cloud2 = random_test(data, row_idx, col_idx)       
        
        return point_cloud1, point_cloud2