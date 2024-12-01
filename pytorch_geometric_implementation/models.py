import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool
from torch_geometric.nn.pool import graclus
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from utils import coarsen_graclus

class GraphChebNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, K=3):
        super(GraphChebNet, self).__init__()
        self.conv1 = ChebConv(num_node_features, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # ChebConv layer + activation ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling to get graph representation
        x = global_mean_pool(x, batch)

        # linear layer for classification
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class GraphChebNetWithCoarsening(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, K=3):
        super(GraphChebNetWithCoarsening, self).__init__()
        self.conv1 = ChebConv(num_node_features, hidden_channels, K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, K)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        edge_index, x, clusters_idx = coarsen_graclus(x, edge_index)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch[clusters_idx])
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


class BaselineMLP(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes):
        super(BaselineMLP, self).__init__()
        self.fc1 = torch.nn.Linear(num_node_features, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        # Global aggregation of nodes feature
        x = global_mean_pool(data.x, data.batch)
        # Option : Comnine with max pooling
        # x_max = global_max_pool(data.x, data.batch)
        # x = torch.cat([x, x_max], dim=1)

        # 3 layers-MLP
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)