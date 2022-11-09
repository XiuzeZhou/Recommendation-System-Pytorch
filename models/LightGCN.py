import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class LightGCN(nn.Module):
    def __init__(self, users_num: int, items_num: int, embedding_size: int, edge_index: torch.tensor):
        super(LightGCN, self).__init__()
        '''
        users_num: number of users;
        items_num: number of items;
        embedding_size: feature size;
        edge_index: edge of graph between user and item;
        '''
        self.edge_index = edge_index
        self.users_num, self.items_num = users_num, items_num
        self.embedding_user = nn.Embedding(self.users_num, embedding_size)
        self.embedding_item = nn.Embedding(self.items_num, embedding_size)
        hidden = embedding_size // 2
        output_size = hidden // 2
        self.conv1 = GCNConv(embedding_size, hidden)
        self.conv2 = GCNConv(hidden, output_size)
        
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embedding_user.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0.1)

    def forward(self, x):
        user_ids, item_ids = x[:,0], x[:,1]
        # graph encoding
        x = torch.cat([self.embedding_user.weight, self.embedding_item.weight], 0)
        x = self.conv1(x, self.edge_index) 
        x = self.conv2(x, self.edge_index)
        #x = F.dropout(x, training=self.training)
        
        users, items = torch.split(x, [self.users_num, self.items_num])
        graph_user, graph_item = users[user_ids], items[item_ids]
        x = torch.sum(graph_user * graph_item, axis=1)

        return x.view(-1)