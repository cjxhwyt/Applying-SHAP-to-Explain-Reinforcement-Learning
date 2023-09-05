# Compare auction types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import pickle

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors[2:])

for filename in os.listdir('../shap'):
# for filename in os.listdir('../shap/local/'):
    if filename.endswith('.csv'):
        params = filename.split('_')
        # comparing FP, SP
        if params[0] == 'permutation' and params[2] == 'DR' and params[3] == '8f' and params[4] == 'default' and params[5] == 'False.csv':
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
            if params[1] == 'FP':
                fp = mean
                fp_txt = txt
            elif params[1] == 'SP':
                sp = mean
                sp_txt = txt

rank = np.argsort(fp)[::-1]
# rank = np.argsort(np.abs(fp))[::-1]
fp, sp = fp[rank], sp[rank]

features = ['Feature ' + str(i + 1) for i in rank]

num = range(1, 9)

fig, ax = plt.subplots()

# Plot SHAP values for each utility estimator
plt.scatter(fp, num, marker='o', label='FP')
plt.scatter(sp, num, marker='*', label='SP')

plt.xlabel('Absolute Mean SHAP Value')
# plt.xlabel('Local SHAP Value')
plt.gca().invert_yaxis()
plt.yticks(ticks=num, labels=features)
plt.legend()
plt.grid(linestyle='dashed', axis='y')

# Define the text
text_lines = [
    ("Average Model Prediction: ", "black"),
    # ("Model Prediction: ", "black"),
    (fp_txt + "(FP)", "#2ca02c"),
    ("/", "black"),
    (sp_txt + "(SP)", "#d62728"),
]

# Calculate the x position for each text
x_pos = [0, 0.43, 0.58, 0.6]  # Adjust the positions as needed
# x_pos = [0, 0.3, 0.45, 0.47]  # Adjust the positions as needed
# Add a text box for each line
y_pos = 1.04
for i, (text, color) in enumerate(text_lines):
    weight = 'bold' if i == 0 else 'normal'  # Make line 0 bold
    ax.text(x_pos[i], y_pos, text, color=color, transform=ax.transAxes, weight=weight)

# plt.show()
plt.savefig('../figures/auction_SHAP.pdf')
# plt.savefig('../figures/auction_SHAP_local.pdf')
