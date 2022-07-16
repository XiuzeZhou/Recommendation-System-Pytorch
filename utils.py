import os
import torch
import torch.nn as nn
import random
import numpy as np
import scipy.sparse as sp


class Dataset(object):
    def __init__(self, path):
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
        


# BPR loss function
class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, y_pos, y_neg):
        distance = y_pos - y_neg
        loss = torch.sum(torch.log((1 + torch.exp(-distance))))
        # loss = torch.sum(-torch.log(self.sigmoid(distance)))
        return loss


# Set a seed for training
def setup_seed(seed):
    np.random.seed(seed)                         # Numpy module.
    random.seed(seed)                            # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)                      # CPU seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)             # GPU
        torch.cuda.manual_seed_all(seed)         # if you are using multi-GPU
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# Read original records
def load_data(file_dir):
    # output: 
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0 
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()
    
        if int(u) not in user_ids_dict:
            user_ids_dict[int(u)]=u_idx
            u_idx+=1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)]=i_idx
            i_idx+=1
        data.append([user_ids_dict[int(u)],rated_item_ids_dict[int(i)],float(r)])
    
    f.close()
    N = u_idx
    M = i_idx

    return N, M, data


# generate pair instances: [user, item, rating]
# generate triple instances: [user, item_0, item_1, rating]
def generate_instances(train_mat, positive_size=1, negative_time=4, is_sparse=False, is_pair=True):
    data = []
    users_num, items_num = train_mat.shape
    
    if is_sparse:
        indptr = train_mat.indptr
        indices = train_mat.indices
    
    if is_pair:    
        for u in range(users_num):
            if is_sparse:
                rated_items = indices[indptr[u]:indptr[u+1]] #用户u中有评分项的id
            else:
                rated_items = np.where(train_mat[u,:]>0)[0]
        
            for i in rated_items:
                data.append([u,i,1.])
                for _ in range(negative_time):
                    j = np.random.randint(items_num)
                    while j in rated_items:
                        j = np.random.randint(items_num)
                    data.append([u,j,0.])
    else:            
        for u in range(users_num):
            if is_sparse:
                rated_items = indices[indptr[u]:indptr[u+1]] #用户u中有评分项的id
            else:
                rated_items = np.where(train_mat[u,:]>0)[0]
        
            for item0 in rated_items:
                for item1 in np.random.choice(rated_items, size=positive_size):
                    data.append([u,item0,item1,1.])
                for _ in range(positive_size*negative_time):
                    item1 = np.random.randint(items_num) # no matter item1 is positive or negtive
                    item2 = np.random.randint(items_num)
                    while item2 in rated_items:
                        item2 = np.random.randint(items_num)
                    data.append([u,item2,item1,0.])
    return data


# generate pairs for BPR: [user, item_0, item_1]
def generate_pairs(train_mat, negative_time=1):
    data = []
    users_num, items_num = train_mat.shape
        
    for u in range(users_num):
        rated_items = np.where(train_mat[u,:] > 0)[0]
        unrated_items = np.where(train_mat[u,:] == 0)[0]
    
        for item0 in rated_items:
            for item1 in np.random.choice(unrated_items, size=negative_time):
                data.append([u, item0, u, item1])

    return data


# Convert the list data to array
def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item interaction matrix
    records_array = np.array(sequence)
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat = np.zeros([N,M])
    mat[row,col] = values
    
    return mat


# Generate train and test from raw user-item interactions
def get_train_test(rating_mat, num_negative=99):
    N, M = rating_mat.shape
    
    selected_items,rest_ratings,negative_items = [],[],[]
    for user_line in rating_mat:
        rated_items = np.where(user_line>0)[0]
        rated_num = len(rated_items)
        random_ids = [i for i in range(rated_num)]
        np.random.shuffle(random_ids)
        selected_id = random_ids[0]
        selected_items.append(rated_items[selected_id])
        rest_ratings.append(rated_items[random_ids[1:]])
        
        unrated_items = np.where(user_line==0)[0]         # ids of unrated items 
        unrated_num = len(unrated_items)                  # number of unrated items 
        random_ids = [i for i in range(unrated_num)]
        np.random.shuffle(random_ids)
        negative_items.append(unrated_items[random_ids[:num_negative]])
        
    train = [[user, item, rating_mat[user,item]] for user in range(N) for item in rest_ratings[user]]   
    test = [[user, selected_items[user]] for user in range(N)]
    
    length = int(N*0.1)
    rated_size = np.sum(rating_mat>0,1)
    rated_order = np.argsort(rated_size)
    sparse_user = rated_order[:length]
    
    np.random.shuffle(train)  
    return train,test,negative_items,sparse_user


# Genarate instances from train dataset
def generate_list(train_mat, negative_time=4):
    data = []
    num_users,_ = train_mat.shape
    
    for u in range(num_users):
        rated_items = np.where(train_mat[u,:]>0)[0]
        unrated_items = np.where(train_mat[u,:]==0)[0]
        
        for i in rated_items:
            data.append([u,i,1.])
            negative_set = np.random.choice(unrated_items, size=negative_time, replace=False)
            for j in negative_set:
                data.append([u,j,0.])
                
    return data


# Read train and test data
def read_data(file_dir):
    data=[]
    f = open(file_dir)
    for line in f.readlines():
        if len(line.split()) == 4:
            u, i, r, _ = line.split()
            data.append([int(u), int(i), float(r)])
        elif len(line.split()) == 3:
            data.append([int(i) for i in line.split()[1:]])
        else:
            u, i = line.split()
            data.append([int(u), int(i)])
    
    return data

















