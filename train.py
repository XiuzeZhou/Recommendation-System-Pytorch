import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from utils import generate_instances
from evaluation import evaluate


def train(model, lr, weight_decay, test_ratings, test_negatives, topK, mat, epochs, batch_size, mode, device):    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss() #nn.MSELoss() nn.BCEWithLogitsLoss()

    hr_list, ndcg_list = [], []
    hr, ndcg = evaluate(model, test_ratings, test_negatives, topK, device=device)
    hr_list.append(hr)
    ndcg_list.append(ndcg)
    print('Init: HR = %.4f, NDCG = %.4f' %(hr, ndcg))
    best_hr, best_ndcg = hr, ndcg

    for epoch in range(epochs):
        model.train()
        data_sequence = generate_instances(mat, negative_time=4)
        data_array = np.array(data_sequence)
        x = torch.from_numpy(data_array[:,:2]).long()
        y = torch.from_numpy(data_array[:,-1])
        data_loader = DataLoader(dataset=TensorDataset(x, y), batch_size=batch_size, shuffle=True)
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = criterion(y_.float(), y.float())
            optimizer.zero_grad()              # clear gradients for this training step
            loss.backward()                    # backpropagation, compute gradients
            optimizer.step()                   # apply gradients

        # Evaluation
        hr, ndcg = evaluate(model, test_ratings, test_negatives, topK, device=device)
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        print('epoch=%d, loss=%.4f, HR=%.4f, NDCG=%.4f' %(epoch, loss, hr, ndcg))

        mlist = hr_list
        if mode == 'ndcg':
            mlist = ndcg_list
        if (len(mlist) > 4) and (mlist[-2] <= mlist[-3] >= mlist[-1]):
            best_hr, best_ndcg = hr_list[-3], ndcg_list[-3]
            break
        best_hr, best_ndcg = hr, ndcg

    print("End. Best HR = %.4f, NDCG = %.4f. " %(best_hr, best_ndcg))
    print('-' * 100)
    return best_hr, best_ndcg