import torch
import torch.nn as nn
import numpy as np


class ConvNCF(torch.nn.Module):
    def __init__(self, users_num, items_num, embedding_size, num_kernel=32, device='cuda'):
        super(ConvNCF, self).__init__()
        self.embedding_size, self.num_kernel, self.device = embedding_size, num_kernel, device
        self.embedding_user, self.embedding_item = nn.Embedding(users_num, embedding_size), nn.Embedding(items_num, embedding_size)
        
        self.cnn = CNNnet(input_size=embedding_size, num_kernel=num_kernel)
        self.linear = nn.Linear(num_kernel, 1, bias=False)
        
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embedding_item.weight, mean=0, std=0.01)
        nn.init.normal_(self.embedding_user.weight, mean=0, std=0.01)
    
    '''    
    def outer(self, vec1, vec2):
        batch_size = vec1.shape[0]
        out = torch.zeros([batch_size, 1, self.embedding_size, self.embedding_size], dtype=torch.float, device=self.device)
        for i in range(batch_size):
            out[i] = torch.outer(vec1[i], vec2[i])
        return out
    '''

    def outer(self, vec1, vec2):
        k1, k2 = vec1.shape[-1], vec2.shape[-1]
        mat1 = vec1.repeat(1, k2).reshape(-1, k2, k1)
        mat1 = torch.transpose(mat1, 1, 2)
        mat2 = vec2.repeat(1, k1).reshape(-1, k1, k2)
        out = mat1 * mat2
        return out

    def forward(self, x):
        user_ids, item_ids = x[:,0], x[:,1]
        embed_users, embed_items = self.embedding_user(user_ids), self.embedding_item(item_ids)
        
        # out = self.outer(embed_users, embed_items)
        out = torch.bmm(embed_users.unsqueeze(2), embed_items.unsqueeze(1))
        out = out.reshape(-1, 1, self.embedding_size, self.embedding_size)
        out = self.cnn(out)
        out = out.reshape(-1, self.num_kernel)
        
        out = self.linear(out)

        return out.view(-1)


class CNNnet(nn.Module):
    def __init__(self, input_size=64, num_kernel=32, kernel_size=2, stride=2, padding=0):
        super(CNNnet, self).__init__()
        num_layers = int(np.log2(input_size))
        self.linears = nn.Sequential()
        for i in range(num_layers):
            conv2d = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
            if i == 0: conv2d = nn.Conv2d(in_channels=1, out_channels=num_kernel, kernel_size=kernel_size, stride=stride, padding=padding)
            self.linears.add_module("cnn_" + str(i), conv2d)
            self.linears.add_module("relu_" + str(i), nn.ReLU())
 
    def forward(self, x):
        out = self.linears(x)
        return out