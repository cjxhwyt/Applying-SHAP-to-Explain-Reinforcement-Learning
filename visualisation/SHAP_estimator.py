# Compare SHAP value estimators
import pandas as pd
import numpy as np

print('4f')
exact_4f = pd.read_csv('../shap/exact_FP_DR_4f_default_False.csv', header=None).to_numpy()
sampling_4f = pd.read_csv('../shap/sampling_FP_DR_4f_default_False.csv', header=None).to_numpy()
partition_4f = pd.read_csv('../shap/partition_FP_DR_4f_default_False.csv', header=None).to_numpy()
kernel_4f = pd.read_csv('../shap/kernel_FP_DR_4f_default_False.csv', header=None).to_numpy()
permutation_4f = pd.read_csv('../shap/permutation_FP_DR_4f_default_False.csv', header=None).to_numpy()

print(f'Exact vs Sampling: {100 - np.mean(np.abs(exact_4f - sampling_4f)) / np.mean(np.abs(exact_4f)) * 100:.2f}%')
print(f'Exact vs Partition: {100 - np.mean(np.abs(exact_4f - partition_4f)) / np.mean(np.abs(exact_4f)) * 100:.2f}%')
print(f'Exact vs Kernel: {100 - np.mean(np.abs(exact_4f - kernel_4f)) / np.mean(np.abs(exact_4f)) * 100:.2f}%')
print(f'Exact vs Permutation: {100 - np.mean(np.abs(exact_4f - permutation_4f)) / np.mean(np.abs(exact_4f)) * 100:.2f}%')

print('-----')

print('8f')
exact_8f = pd.read_csv('../shap/exact_FP_DR_8f_default_False.csv', header=None).to_numpy()
sampling_8f = pd.read_csv('../shap/sampling_FP_DR_8f_default_False.csv', header=None).to_numpy()
partition_8f = pd.read_csv('../shap/partition_FP_DR_8f_default_False.csv', header=None).to_numpy()
kernel_8f = pd.read_csv('../shap/kernel_FP_DR_8f_default_False.csv', header=None).to_numpy()
permutation_8f = pd.read_csv('../shap/permutation_FP_DR_8f_default_False.csv', header=None).to_numpy()

print(f'Exact vs Sampling: {100 - np.mean(np.abs(exact_8f - sampling_8f)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
print(f'Exact vs Partition: {100 - np.mean(np.abs(exact_8f - partition_8f)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
print(f'Exact vs Kernel: {100 - np.mean(np.abs(exact_8f - kernel_8f)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
print(f'Exact vs Permutation: {100 - np.mean(np.abs(exact_8f - permutation_8f)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')

print('-----')

print('kernel')
kernel_100 = pd.read_csv('../shap/kernel_100_FP_DR_8f_default_False.csv', header=None).to_numpy()
kernel_kmeans = pd.read_csv('../shap/kernel_100_kmeans_FP_DR_8f_default_False.csv', header=None).to_numpy()
print(f'Exact vs Kernel 800: {100 - np.mean(np.abs(exact_8f - kernel_8f)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
print(f'Exact vs Kernel 80: {100 - np.mean(np.abs(exact_8f - kernel_100)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
print(f'Exact vs Kernel kmeans: {100 - np.mean(np.abs(exact_8f - kernel_kmeans)) / np.mean(np.abs(exact_8f)) * 100:.2f}%')
