import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange


from cfn_model_generation import *



### Parameters
n_internal_verts = 200 #Number of internal vertices in the tree
batch_size = 10000 #Number of spins per batch.
num_batches = 200 #Number of batches per tree
tree_types = [2, 3, 4, 5,6,7,8] #Degree types for (d+1)-regular rooted trees
trials_per_tree = batch_size * num_batches # Trials per tree
total_trials = trials_per_tree * len(tree_types) # Total number of trials
delta = 1/10  #Value of delta. Theoretical max is delta = 1/924
C_val = .5 #C_{2.1} and C_{2.2}
c_val = 0.25 #c_{2.1} and c_{2.2}
C_upperTail = 4/5 * (9 * C_val**2 / (2*c_val) + 2* C_val)**2 #C_{2.7}
c_lowerTail = 1 - 2*c_val/(3*C_val) #c_{2.8}
probability_upper_tail_constant = 7*C_val #c_{2.7}
probability_lower_tail_constant = 78*C_val**2 #C_{2.8}
upper_tail_sample_threshold = 1 - C_upperTail* delta**2
lower_tail_sample_threshold = -c_lowerTail
upper_tail_probability = 1 - probability_upper_tail_constant*delta
lower_tail_probability = probability_lower_tail_constant * delta**2
print(f"The theoretical upper-tail bound is {upper_tail_probability:.4} and lower-tail bound is {lower_tail_probability:.4} for delta = {delta}.")
num_bins = max(500,int(10/delta))



### Generating Histograms
bins = np.linspace(-1,1, num_bins+1)
histograms = [np.histogram([], bins)[0] for _ in tree_types]
empircal_count_above_threshold = []
empircal_count_below_threshold = []
for ind, deg in enumerate(tree_types):
    print(f'Working on tree {ind+1}/{len(tree_types)}.')
    empircal_count_above_threshold.append(0)
    empircal_count_below_threshold.append(0)
    Z = histograms[ind]
    deg_sequence = [deg for _ in range(n_internal_verts)]
    root = GeneralizedFoataFuchs(deg_sequence, delta, C_val = C_val, c_val = c_val)
    for i in trange(num_batches):
        attempt = root.constructMagnetization(batch_size)
        root.magnetizations = None

        empircal_count_above_threshold[-1] += sum(attempt >= upper_tail_sample_threshold)
        empircal_count_below_threshold[-1] += sum(attempt <= lower_tail_sample_threshold)

        binned_attempt = np.histogram(attempt, bins)[0]

        Z = Z+binned_attempt
    histograms[ind] = Z
    p_up = empircal_count_above_threshold[-1]/trials_per_tree
    p_down = empircal_count_below_threshold[-1]/trials_per_tree
    print(f"Proportion of samples above upper threshold {p_up}.")
    print(f"Proportion of samples above upper threshold {p_down}.")

### Saving Histograms
df = dict()
df['left_end_point'] = list(bins[:-1])
for counts, deg in zip(histograms, tree_types):
    df['histogram_for_'+str(deg+1)+'-regular_tree'] = list(counts)

dataframe = pd.DataFrame(df)
dataframe.to_csv('../data/magnetization/data.csv')

