import torch
import math
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


######################################################################
################# evaluate by negative sampling ######################
######################################################################
def evaluate(model, test_ratings, test_negatives, topK, is_pair=True, device='cpu'):
    model.eval()
    pairs_list = get_pairs(test_ratings, test_negatives)
    ranklist = []
    with torch.no_grad():
        for pairs in pairs_list:
            pairs = np.array(pairs)
            items = pairs[:,1]
        
            x = pairs
            x = torch.from_numpy(x).long()
            if not is_pair: x = torch.cat([x, torch.from_numpy(np.array(items)).long().reshape(-1, 1)], 1)
            x = x.to(device)
            out = model(x)
            scores = out.reshape(-1).detach().cpu().numpy()
        
            scores_sorted = (np.array(scores)).argsort()[::-1]    # 倒序排列
            predictions = [items[i] for i in scores_sorted[:topK]]  # 取前 TopK 个
            ranklist.append(predictions)
        hr, ndcg = hr_ndcg(np.array(test_ratings), ranklist)
    return hr, ndcg


# combine test_ratings and test_negatives to pairs:(u,i)
def get_pairs(test_ratings, test_negatives):
    assert len(test_ratings) == len(test_negatives)
    pairs = []
    N = len(test_ratings)                           # N users
    for i in range(N):
        user, item_test = test_ratings[i][0], test_ratings[i][1]
        pair = [test_ratings[i]]
        for j in test_negatives[i]: pair.append([user, j])
        pairs.append(pair)
    return pairs


def get_hit(ranklist, rated_item):
    result = 0
    for item in ranklist:
        if item == rated_item:
            result = 1
    return result
    
    
def get_ndcg(ranklist, rated_item):
    result = 0
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == rated_item:
            result = math.log(2)/math.log(i+2)
    return result


def hr_ndcg(test_sequence, ranklist):
    length = len(test_sequence)
    hits, ndcgs=[], []
    for idx in range(length):
        user = test_sequence[idx,0].astype(np.int32)
        rated_item = test_sequence[idx,1].astype(np.int32)
        hit = get_hit(ranklist[user], rated_item)
        ndcg = get_ndcg(ranklist[user], rated_item)
        hits.append(hit)
        ndcgs.append(ndcg)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg


# get top_k list from test sequence
def get_topk(prediction, test_sequence, topK=10):
    assert len(prediction)==len(test_sequence)
    users_num = len(prediction)
    items_unrated = test_sequence[:,1:] # [rated_item, negative_item1,...]
    unrated_item_scores = np.array([prediction[u,items_unrated[u,:]] for u in range(users_num)])
    y_pred_order = np.argsort(-unrated_item_scores)
    topk = np.array([items_unrated[u,y_pred_order[u,:]] for u in range(users_num)])

    return topk[:,:topK]

######################################################################
###################### evaluate by matrix ############################
######################################################################
# get top_n list from train matrix
def get_topn(r_pred, train_mat, n=10):
    unrated_items = r_pred * (train_mat==0)
    idx = np.argsort(-unrated_items)
    return idx[:,:n]


def recall_precision(topn, test_mat):
    n,m = test_mat.shape
    hits,total_pred,total_true = 0.,0.,0.
    for u in range(n):
        hits += len([i for i in topn[u,:] if test_mat[u,i]>0])
        size_pred = len(topn[u,:])
        size_true = np.sum(test_mat[u,:]>0,axis=0)
        total_pred += size_pred
        total_true += size_true

    recall = hits/total_true
    precision = hits/total_pred
    return recall, precision


def mae_rmse(r_pred, test_mat):
    y_pred = r_pred[test_mat>0]
    y_true = test_mat[test_mat>0]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse 


# evaluate model by matrix
def evaluation(pred_mat, train_mat, test_mat):
    topn = get_topn(pred_mat, train_mat, n=10)
    mae, rmse = mae_rmse(pred_mat, test_mat)
    recall, precision = recall_precision(topn, test_mat)
    return mae, rmse, recall, precision