# Compare utility estimators
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

for filename in os.listdir('../shap'):
# for filename in os.listdir('../shap/local/'):
    if filename.endswith('.csv'):
        params = filename.split('_')
        # comparing DM, DR, IPS
        if params[0] == 'permutation' and params[1] == 'FP' and params[3] == '8f' and params[4] == 'default' and params[5] == 'False.csv':
            temp = pd.read_csv('../shap/' + filename, header=None).to_numpy()
            # temp = pd.read_csv('../shap/local/' + filename, header=None).to_numpy()
            mean = np.mean(np.abs(temp), axis=0)
            # mean = temp[0]
            with open('../shap/' + filename[:-4] + '.pkl', 'rb') as file:
                shap = pickle.load(file)
            txt = '%.3f' % shap.base_values[0]
            # with open('../shap/local/' + filename[:-4] + '.pkl', 'rb') as file:
            #     shap = pickle.load(file)
            # txt = '%.3f' % float((np.sum(shap.values) + shap.base_values))
            print(txt)
            if params[2] == 'DM':
                dm = mean
                dm_txt = txt
            elif params[2] == 'IPS':
                ips = mean
                ips_txt = txt
            elif params[2] == 'DR':
                dr = mean
                dr_txt = txt

rank = np.argsort(dr)[::-1]
# rank = np.argsort(np.abs(dr))[::-1]
dm, ips, dr = dm[rank], ips[rank], dr[rank]

features = ['Feature ' + str(i + 1) for i in rank]

num = range(1, 9)

fig, ax = plt.subplots()

# Plot SHAP values for each utility estimator
plt.scatter(dm, num, marker='^', label='DM')
plt.scatter(ips, num, marker='s', label='IPS')
plt.scatter(dr, num, marker='o', label='DR')

# plt.xlabel('Absolute Mean SHAP Value')
plt.xlabel('Local SHAP Value')
plt.gca().invert_yaxis()
plt.yticks(ticks=num, labels=features)
plt.legend()
plt.grid(linestyle='dashed', axis='y')

# Define the text
text_lines = [
    ("Average Model Prediction: ", "black"),
    # ("Model Prediction: ", "black"),
    (dm_txt + "(DM)", "#1f77b4"),
    ("/", "black"),
    (ips_txt + "(IPS)", "#ff7f0e"),
    ("/", "black"),
    (dr_txt + "(DR)", "#2ca02c")
]

# Calculate the x position for each text
x_pos = [0, 0.43, 0.59, 0.61, 0.77, 0.79]  # Adjust the positions as needed
# Add a text box for each line
y_pos = 1.04
for i, (text, color) in enumerate(text_lines):
    weight = 'bold' if i == 0 else 'normal'  # Make line 0 bold
    ax.text(x_pos[i], y_pos, text, color=color, transform=ax.transAxes, weight=weight)

# plt.show()
plt.savefig('../figures/utility_SHAP.pdf')
# plt.savefig('../figures/utility_SHAP_local.pdf')
