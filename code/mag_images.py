import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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

## Load data
df = pd.read_csv('../data/magnetization/data.csv')



bins = list(df['left_end_point'])
bins.append(1)
bins = np.array(bins)

degrees = list(range(2,9))
hist_columns = list(df.columns)
hist_columns = hist_columns[3:]

histograms = [np.array(df[c]) for c in hist_columns]
total_trials = sum(histograms[0])



gap = bins[1]-bins[0]


fig, axes = plt.subplots(7,1, figsize = (10,15), sharex= True)

plt.suptitle(f"Empirical Distribution of $\\sigma_u Z_u$ for $\\delta = ${delta:.2}.")

alpha = 0.25
for ind, X in enumerate(zip(histograms, degrees)):
    Z, deg = X
    smoothed_out_histogram = np.hstack((0,Z[:-1]))+np.hstack((Z[1:], 0)) + 2*Z
    smoothed_out_histogram = smoothed_out_histogram/sum(smoothed_out_histogram)*(1/gap)

    A = max(smoothed_out_histogram)+100
    B = A - 100
    axes[ind].stairs([A],[lower_tail_sample_threshold,upper_tail_sample_threshold], fill = True, color = 'blue', alpha = alpha)
    axes[ind].stairs([A], [upper_tail_sample_threshold,1],fill = True, color = 'g', alpha = alpha)
    axes[ind].stairs([A], [-1,lower_tail_sample_threshold],fill = True, color = 'r', alpha = alpha )
    axes[ind].stairs(smoothed_out_histogram, bins, fill = True, color = 'k')
    axes[ind].annotate('Severe Failure', xy = (-.98,B))
    axes[ind].annotate('Moderate Failure', xy = (-.16,B))
    axes[ind].annotate('Successful', xy = (.775,B))
    if deg==2:
        title = 'Bin'
    elif deg==3:
        title = 'Tern'
    else:
        title = str(deg)+'-'
    title = title + 'ary Tree'
    plt.yscale('log')

    axes[ind].set_title(title)
    axes[ind].set_yscale('log')
plt.show()