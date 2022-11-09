import torch
import torch.nn as nn


class BPR(nn.Module):
    def __init__(self, users_num: int, items_num: int, embedding_size: int = 64):
        super(BPR, self).__init__()
        '''
        users_num: number of users;
        items_num: number of items;
        embedding_size: feature size;
        '''
        self.embed_user = nn.Embedding(users_num, embedding_size)
        self.embed_item = nn.Embedding(items_num, embedding_size)
        
        self._init_weight_()
        
    def _init_weight_(self):
        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        
    def forward(self, x):
        user_ids, item_ids = x[:,0], x[:,1]
        embed_user, embed_item = self.embed_user(user_ids), self.embed_item(item_ids)
        out = torch.sum(embed_user * embed_item, axis=1)
        return out.view(-1)