import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, global_mean_pool

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