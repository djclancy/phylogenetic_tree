import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
import copy
import networkx as nx
from cfn_model_generation import *



### Parameters


delta = 1/100  #Value of delta. Theoretical max is delta = 1/924
C_val = .5 #C_{2.1} and C_{2.2}
c_val = 0.25 #c_{2.1} and c_{2.2}
C_upperTail = 4/5 * (9 * C_val**2 / (2*c_val) + 2* C_val)**2 #C_{2.7}
c_lowerTail = 1 - 2*c_val/(3*C_val) #c_{2.8}


n_leaves = 10
n_samples = 1000000
errors_coord_update =[]
errors_gradient_ascent = []
edge_sets, vertex_sets, roots = [],[],[]
epochs = 3
learning_rates = [ 0.0003]#, 0.01, 0.005, 0.001]

for lr in learning_rates:
    tree, edges, vertices, root = UniformTree(n_leaves,n_samples, delta, warmStart=True, C_val = C_val, c_val = c_val)
    tree2 = copy.deepcopy(tree)
    edge_sets.append(edges)
    vertex_sets.append(vertices),
    roots.append(root)
    error_coord = np.array([tree.get_gaps([1,2])]).reshape(-1,1)
    error_grad =  np.array([tree2.get_gaps([1,2])]).reshape(-1,1)
    print('Coordinate Update')
    for i in range(epochs):
        next_round_coord = np.array(tree.update_round([1,2]))
        error_coord = np.hstack((error_coord, next_round_coord))
    print('Gradient Ascent')
    for i in range(epochs):
        next_round_grad = np.array(tree2.update_round([1,2], update_rule='gradient', learning_rate = lr))
        error_grad = np.hstack((error_grad, next_round_grad))
    errors_coord_update.append(error_coord)
    errors_gradient_ascent.append(error_grad)



num_edges = n_leaves * 2 - 1

t = list(range(epochs*num_edges+1))
t1 = [1+k*num_edges for k in range(epochs+1)]


fig, ax = plt.subplots(1,3,figsize = (18,5), 
                       gridspec_kw={'width_ratios': [1,1, 1.5]})
A = .25
alpha = .25
for i in range(2):
    for s in t1:
        ax[i].plot([s,s],[0,A], color = 'k', alpha=alpha)        

linewidth = .75
m = 0
M = float('inf')
for i, e in enumerate(errors_coord_update):
    m = min([min(e[1,:]/np.sqrt(num_edges)), min(e[0,:]/num_edges), m ])
    M = max([max(e[1,:]/np.sqrt(num_edges)), max(e[0,:]/num_edges), M ])
    if i==0:
        ax[0].plot(t, e[1,:]/np.sqrt(num_edges), 'k', label = 'RMSE', linewidth = linewidth)
        ax[0].plot(t, e[0,:]/num_edges, 'r', label = 'MAE', linewidth = linewidth)
    else:
        ax[0].plot(t, e[1,:]/np.sqrt(num_edges), 'k', linewidth = linewidth)
        ax[0].plot(t, e[0,:]/num_edges, 'r', linewidth = linewidth)
for i, e in enumerate(errors_gradient_ascent):
    m = min([min(e[1,:]/np.sqrt(num_edges)), min(e[0,:]/num_edges), m ])
    M = max([max(e[1,:]/np.sqrt(num_edges)), max(e[0,:]/num_edges), M ])
    if i==0:
        ax[1].plot(t, e[1,:]/np.sqrt(num_edges), 'k', label = 'RMSE', linewidth = linewidth)
        ax[1].plot(t, e[0,:]/num_edges, 'r', label = 'MAE', linewidth = linewidth)
    else:
        ax[1].plot(t, e[1,:]/np.sqrt(num_edges), 'k', linewidth = linewidth)
        ax[1].plot(t, e[0,:]/num_edges, 'r', linewidth = linewidth)

ax[0].plot(t, [delta for _  in t],'k', ls = '-.', label = '$\\delta$')
ax[0].plot(t, [delta**2 for _  in t], 'k--',label = '$\\delta^2$')
ax[1].plot(t, [delta for _  in t],'k', ls = '-.', label = '$\\delta$')
ax[1].plot(t, [delta**2 for _  in t], 'k--',label = '$\\delta^2$')
ax[0].legend(loc = (.05,.1))
ax[1].legend(loc = (0.05, .1))

ax[0].set_yscale('log')
ax[1].set_yscale('log')

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
ax[2] = nx.draw(graph, pos = pos, node_size=100)

plt.show()