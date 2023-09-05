# Compare intermediate models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

folder = 'default'
iterations = np.arange(0, 50, 3)

shap = pd.DataFrame(columns=['Feature ' + str(i+1) for i in range(8)])
for i in iterations:
    data = pd.read_csv('../shap/' + folder + '/iter' + str(i) + '_FP_DR_8f_' + folder + '_False.csv', header=None).to_numpy()
    abs_mean = pd.Series(np.mean(np.abs(data), axis=0), index=shap.columns)
    shap = shap.append(abs_mean, ignore_index=True)
shap = shap.rename(index=lambda x: 3 * x)
print(shap)

shap_2 = pd.DataFrame(columns=['Feature ' + str(i+1) for i in range(8)])
for i in iterations:
    data = pd.read_csv('../shap/local/' + folder + '/permutation_iter' + str(i) + '_FP_DR_8f_' + folder + '_False.csv', header=None).to_numpy()
    abs_mean = pd.Series(data[0], index=shap_2.columns)
    shap_2 = shap_2.append(abs_mean, ignore_index=True)
shap_2 = shap_2.rename(index=lambda x: 3 * x)
print(shap_2)

bidder_surplus = pd.read_csv('../training/progressTrue_FP_DR_8f_' + folder + '_False_train_info.csv')['my_surplus'][::3].rename('Bidder Surplus')
print(bidder_surplus)

model_average = []
for i in iterations:
    with open('../shap/' + folder + '/iter' + str(i) + '_FP_DR_8f_default_False.pkl', 'rb') as file:
        temp = pickle.load(file)
    model_average.append(round(temp.base_values[0], 3))

# Create a figure with four subplots
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(16, 9))

# Plot bidder surplus in the first subplot
axs[0].plot(iterations, bidder_surplus, marker='o', markersize=2, label='Bidder Surplus', color='#2ca02c')

axs[0].xaxis.remove_overlapping_locs = False
axs[0].set_ylabel('Bidder Surplus')
axs[0].legend(loc=4)
axs[0].set_xticks(np.arange(0, 50, 10))
axs[0].set_xticks(np.arange(0, 50, 3), minor=True)
axs[0].grid(linestyle='dashed', axis='x', which='minor')

# Plot model average prediction in the second subplot
axs[1].plot(iterations, model_average, marker='o', markersize=2, label='Average Model Prediction', color='#17becf')

axs[1].xaxis.remove_overlapping_locs = False
axs[1].set_ylim(0.7, 1.3, 0.2)
axs[1].set_ylabel('Average \nModel Prediction')
axs[1].legend()
axs[1].set_xticks(np.arange(0, 50, 10))
axs[1].set_xticks(np.arange(0, 50, 3), minor=True)
axs[1].grid(linestyle='dashed', axis='x', which='minor')

colours = ['#1f77b4', '#ff7f0e', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
# Plot absolute mean SHAP values for each feature in the third subplot
for feature_idx in range(1, 9):  # Assuming columns are named 'Feature 1' to 'Feature 8'
    axs[2].plot(shap.index, shap[f'Feature {feature_idx}'], color=colours[feature_idx-1], label=f'Feature {feature_idx}', marker='o', markersize=2)

axs[2].xaxis.remove_overlapping_locs = False
axs[2].set_ylabel('Absolute Mean \nSHAP Value')
axs[2].legend(loc=1, ncol=4)
axs[2].set_xticks(np.arange(0, 50, 10))
axs[2].set_xticks(np.arange(0, 50, 3), minor=True)
axs[2].grid(linestyle='dashed', axis='x', which='minor')

# Plot SHAP value (an instance) for each feature in the forth subplot
for feature_idx in range(1, 9):  # Assuming columns are named 'Feature 1' to 'Feature 8'
    axs[3].plot(shap_2.index, shap_2[f'Feature {feature_idx}'], color=colours[feature_idx-1], label=f'Feature {feature_idx}', marker='o', markersize=2)

axs[3].xaxis.remove_overlapping_locs = False
axs[3].set_xlabel('Iteration')
axs[3].set_ylabel('Local SHAP Value')
axs[3].legend(loc=1, ncol=4)
axs[3].set_xticks(np.arange(0, 50, 10))
axs[3].set_xticks(np.arange(0, 50, 3), minor=True)
axs[3].grid(linestyle='dashed', axis='x', which='minor')

# plt.tight_layout()
# plt.show()
plt.savefig('../figures/progress.pdf')
