### Updated version of coordinate_update.py


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
import copy
import networkx as nx
from cfn_model_generation import *
import json
import datetime

file_name = '../data/coord_update_and_tree/'+str(datetime.datetime.now().strftime("%d%m%Y%H%M"))


### Parameters


delta = 1/10  #Value of delta. Theoretical max is delta = 1/924
C_val = .5 #C_{2.1} and C_{2.2}
c_val = 0.25 #c_{2.1} and c_{2.2}
C_upperTail = 4/5 * (9 * C_val**2 / (2*c_val) + 2* C_val)**2 #C_{2.7}
c_lowerTail = 1 - 2*c_val/(3*C_val) #c_{2.8}


n_leaves = 10
num_edges = n_leaves * 2 - 1
n_trees = 1
n_samples = 1000000 #200000
errors_coord_update =[]
errors_gradient_ascent = []
error_empirical_data = []
edge_sets, vertex_sets, roots = [],[],[]
epochs = 5
lr = 0.035#, 0.01, 0.005, 0.001]
warm_start = False
track = True

scale_mat = np.diag([1/num_edges, 1/np.sqrt(num_edges)])
for _ in trange(n_trees):
    #print(f"Round {_+1}/{n_trees}:")
    if _==0:
        tree, edges, vertices, root = UniformTree(n_leaves,n_samples, delta, warmStart=warm_start, C_val = C_val, c_val = c_val)
    else:
        tree = Tree(vertices, edges, root, n_samples, delta, warmStart=warm_start, C_val=C_val, c_val=c_val)
    tree2 = copy.deepcopy(tree)
    #tree3 = copy.deepcopy(tree)
    edge_sets.append(edges)
    vertex_sets.append(vertices),
    roots.append(root)
    error_coord = np.array([tree.get_gaps([1,2])]).reshape(-1,1) 
    error_grad =  np.array([tree2.get_gaps([1,2])]).reshape(-1,1) 
    
    #error_emp = np.array([tree3.get_gaps([1,2])]).reshape(-1,1)
    #print('Coordinate Update')
    for i in range(epochs):
        next_round_coord = np.array(tree.update_round([1,2], print_progress=track))
        error_coord = np.hstack((error_coord, next_round_coord))
    #print('Gradient Ascent')
    for i in range(epochs):
        next_round_grad = np.array(tree2.update_round([1,2], update_rule='gradient', learning_rate = lr, print_progress=track))
        error_grad = np.hstack((error_grad, next_round_grad))

    
    errors_coord_update.append(scale_mat@error_coord)
    errors_gradient_ascent.append(scale_mat@error_grad)
    #error_empirical_data.append(error_emp)


edges_in_graph = list(edges.values())


t = list(range(epochs*num_edges+1))
t1 = [1+k*num_edges for k in range(epochs+1)]



coord_update_array = np.array(errors_coord_update)
grad_update_array = np.array(errors_gradient_ascent)

np.save(file_name+'_coord',coord_update_array)
np.save(file_name+'_grad',grad_update_array)
np.save(file_name+'__edges', edges_in_graph)


coord_update_mean = np.mean(coord_update_array,0)
coord_update_std = np.std(coord_update_array,0)
coord_update_max = np.max(coord_update_array, 0)
coord_update_min = np.min(coord_update_array, 0)

coord_update_upper = coord_update_mean + coord_update_std
coord_update_lower = coord_update_mean - coord_update_std

grad_update_mean = np.mean(grad_update_array,0)
grad_update_std = np.std(grad_update_array,0)
grad_update_max = np.max(grad_update_array, 0)
grad_update_min = np.min(grad_update_array, 0)

grad_update_upper = grad_update_mean + grad_update_std
grad_update_lower = grad_update_mean - grad_update_std







fig, ax = plt.subplots(1,3,figsize = (18,5), 
                       gridspec_kw={'width_ratios': [1,1, 1.15]})
A = .25
alpha = .25
for i in range(2):
    for s in t1:
        ax[i].plot([s,s],[0,A], color = 'k', alpha=alpha)        

linewidth = .75
m = min(np.min(coord_update_array), np.min(grad_update_array))
M = max(np.max(coord_update_array), np.max(grad_update_array))

ax[0].set_title('Coordinate Maximization')
ax[0].fill_between(t, coord_update_max[0,:], coord_update_min[0,:], alpha = alpha, color = 'g')
ax[0].fill_between(t, coord_update_max[1,:], coord_update_min[1,:], alpha = alpha, color = 'r')
ax[0].fill_between(t, coord_update_upper[0,:], coord_update_lower[0,:], alpha = 2*alpha, color = 'g')
ax[0].fill_between(t, coord_update_upper[1,:], coord_update_lower[1,:], alpha = 2*alpha , color = 'r')
ax[0].plot(t, coord_update_mean[0,:], color = 'g', linewidth = linewidth, label = 'MAE')
ax[0].plot(t, coord_update_mean[1,:], color = 'r', linewidth = linewidth, label = 'RMSE')

ax[1].set_title('Cyclic Gradient Ascent')
ax[1].fill_between(t, grad_update_max[0,:], grad_update_min[0,:], alpha = alpha, color = 'g')
ax[1].fill_between(t, grad_update_max[1,:], grad_update_min[1,:], alpha = alpha , color = 'r')
ax[1].fill_between(t, grad_update_upper[0,:], grad_update_lower[0,:], alpha = 2*alpha, color = 'g')
ax[1].fill_between(t, grad_update_upper[1,:], grad_update_lower[1,:], alpha = 2*alpha , color = 'r')
ax[1].plot(t, grad_update_mean[0,:], color = 'g', linewidth = linewidth, label = 'MAE')
ax[1].plot(t, grad_update_mean[1,:], color = 'r', linewidth = linewidth, label = 'RMSE')



for i in range(2):
    ax[i].plot(t, [delta for _  in t],'k', ls = '-.', label = '$\\delta$')
    ax[i].plot(t, [delta**2 for _  in t], 'k--',label = '$\\delta^2$')
    ax[i].legend(loc = (.05,.1))
    ax[i].set_yscale('log')


'''x1,x2,x3 = np.mean(t1), np.mean(t2), np.mean(t3)
offset = 90
x1, x2, x3 = x1-offset, x2-offset, x3-offset
ax[0].annotate('First Round', xy = (x1,A/2))
ax[0].annotate('Second Round', xy = (x2,A/2))
ax[0].annotate('Third Round', xy = (x3,A/2))
ax[1].annotate('First Round', xy = (x1,A/2))
ax[1].annotate('Second Round', xy = (x2,A/2))
ax[1].annotate('Third Round', xy = (x3,A/2))'''
plt.suptitle("Error between $\\theta$ and $\\hat{\\theta}$ without warm start.")
edges = list(edges.values())
graph = nx.Graph(edges)
pos = nx.kamada_kawai_layout(graph)
ax[-1] = nx.draw(graph, pos = pos, node_size=100)

graph

plt.show()
