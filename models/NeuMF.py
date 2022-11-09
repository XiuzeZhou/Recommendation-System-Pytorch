import torch
import torch.nn as nn


class NeuMF(torch.nn.Module):
    def __init__(self, users_num, items_num, embedding_size=16, num_layers=3):
        super(NeuMF, self).__init__()
        '''
        users_num: number of users;
        items_num: number of items;
        embedding_size: feature size;
        num_layers: the number of layers in MLP model;
        '''
        
        self.embedding_size_gmf, self.embedding_size_mlp = embedding_size, embedding_size*(2 ** (num_layers - 1))
        self.embedding_item_gmf = nn.Embedding(items_num, self.embedding_size_gmf)
        self.embedding_user_gmf = nn.Embedding(users_num, self.embedding_size_gmf)
        self.embedding_item_mlp = nn.Embedding(items_num, self.embedding_size_mlp)
        self.embedding_user_mlp = nn.Embedding(users_num, self.embedding_size_mlp)
        
        self.mlp = nn.Sequential()
        input_size = 2 * self.embedding_size_mlp
        for i in range(num_layers):
            output_size = input_size // 2
            self.mlp.add_module('linear' + str(i), nn.Linear(input_size, output_size))
            self.mlp.add_module('relu' + str(i), nn.ReLU())
            input_size = output_size
        
        self.linear = nn.Linear(2 * self.embedding_size_gmf, 1)

        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embedding_item_gmf.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_user_gmf.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_item_mlp.weight, mean=0, std=0.1)
        nn.init.normal_(self.embedding_user_mlp.weight, mean=0, std=0.1)
        
    def forward(self, x):
        user_ids, item_ids = x[:,0], x[:,1]
        embed_items_gmf, embed_users_gmf = self.embedding_item_gmf(item_ids), self.embedding_user_gmf(user_ids)
        
        # GMF
        out_gmf = embed_items_gmf * embed_users_gmf
        
        # MLP
        embed_items_mlp, embed_users_mlp = self.embedding_item_mlp(item_ids), self.embedding_user_mlp(user_ids)
        out = torch.cat([embed_users_mlp, embed_items_mlp], 1).reshape(-1, 2*self.embedding_size_mlp)
        out_mlp = self.mlp(out)
        
        # prediction
        out = torch.cat((out_gmf, out_mlp), -1)
        out = self.linear(out)
        out = out.view(-1)
        return out