import copy
import torch
from torch import nn
import random
import collections
import numpy as np
import operator
import pandas as pd
from scipy import stats

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def krum_median(w,args):
    distances =collections.defaultdict(dict)
    non_malicious_count = int((args.num_users - args.atk_num) * args.frac)
    num = 0
    m = args.num_users - 2 * args.atk_num-1 #the number of the select set,and the number have to smaller than args.num_users - 2 * args.atk_num
    min_dist_set = [0]*args.num_users #save the m smallest distance worker and their index
    min_m_error = []
    min_m_index = []
    w_list = []
    for k in w[0].keys():
        if num == 0:
            for i in range(len(w)):
                for j in range(i):
                    distances[i][j] = distances[j][i] = np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
            num = 1
        else:
            for i in range(len(w)):
                for j in range(i):
                    distances[j][i] += np.linalg.norm(w[i][k].cpu().numpy() - w[j][k].cpu().numpy())
                    distances[i][j] += distances[j][i]
    # print("distance = ",distances)
    # sorted_dist = sorted(distances.items(), key=lambda x: x[1][1])
    # min_dist_set = sorted_dist[0:m]
    # for i in range(m):
    #     min_m_error.append(min_dist_set[i][1])
    #     min_m_index.append(min_dist_set[i][0])
    #     w_list.append(w[min_m_index])
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])
        min_dist_set[user] = current_error
    min_dist_set = pd.Series(min_dist_set).sort_values()
    minimal_error_index = pd.Series(min_dist_set).sort_values().index[:m]
    for i in range(m):
        w_list.append(w[minimal_error_index[i]])
    number_to_consider = int((args.num_users - args.atk_num) * args.frac) - 1
    number_to_consider = m-1
    #print(number_to_consider)


    w_avg = copy.deepcopy(w_list[0])
    for k in w_avg.keys():
        tmp = []
        for i in range(len(w_list)):
            tmp.append(w_list[i][k].cpu().numpy())  # get the weight of k-layer which in each client
        tmp = np.array(tmp)
        #tmp = stats.trimboth(tmp,0.1)
        tmp = random.sample(list(tmp),int(0.8*len(tmp)))
        #tmp = random.sample(tmp,5)
        med = np.median(tmp, axis=0)
        #mean = gmean(tmp,axis=0)
        new_tmp = []
        # for i in range(len(tmp)):  # cal each client weights - median
        #     new_tmp.append(tmp[i] - med)
        new_tmp = np.array(new_tmp)
        # good_vals = np.argsort(abs(new_tmp), axis=0)[:number_to_consider]
        # good_vals = np.take_along_axis(new_tmp, good_vals, axis=0)
        # k_weight = np.array(np.mean(good_vals) + med)
        #w_avg[k] = torch.from_numpy(k_weight).to(args.device)
        w_avg[k] = torch.from_numpy(med).to(args.device)
    #print("wglobal = ",w_avg)
    return w_avg