# (Step3) Generate SHAP values for special experiment settings: k-means for KernelSHAP and explanation of intermediate models
import sys

sys.path.insert(1, '../src')

import argparse
import numpy as np
import pandas as pd
import pickle
import shap
import torch
import matplotlib.pyplot as plt
from main import parse_config
from BidderAllocation import OracleAllocator

import time
from datetime import timedelta

def simulate_a_bid(contexts):
    # prevent from random results
    torch.manual_seed(config['random_seed'])

    # Sample a true context vector
    true_context = contexts

    # Mask true context into observable context
    obs_context = np.concatenate((true_context[:auction_model.obs_embedding_size], [1.0]))

    for agent_id, agent in enumerate(auction_model.agents):
        if agent_id == my_agent_id:
            if isinstance(agent.allocator, OracleAllocator):
                bid, item = agent.bid(true_context)
            else:
                bid, item = agent.bid(obs_context)

    return bid


def simulate_bids_permutation(all_contexts):
    result = []

    for _, contexts in all_contexts.iterrows():
        result.append(simulate_a_bid(contexts))

    return np.array(result)


def simulate_bids(all_contexts):
    result = []

    for contexts in all_contexts:
        result.append(simulate_a_bid(contexts))

    return np.array(result)


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('way', type=str, help='Way to generate data')
    parser.add_argument('specific', type=str, help='Whether to generate specific data')
    parser.add_argument('size', type=int, help='Size of data')
    parser.add_argument('exp', type=str, help='Experiment of either progress or background data')
    parser.add_argument('iter', type=str, help='Loop iterations', nargs='?')
    args = parser.parse_args()

    specific = eval(args.specific)
    data_size = args.size
    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, \
        obs_embedding_size = parse_config('../config/' + args.config)

    my_agent_id = 0
    filename = args.config[:-5] + '_' + args.way + '_' + str(specific)

    with open('../model/' + filename + '.pkl', 'rb') as file:
        auction_model = pickle.load(file)
        print(f'Object successfully load from "{filename}.pkl"')

    X_train = pd.read_csv('../data/' + filename + '_' + str(data_size) + '_X_train.csv')
    X_test = pd.read_csv('../data/' + filename + '_' + str(data_size) + '_X_test.csv')
    X_train_obs = X_train.iloc[:, :auction_model.obs_embedding_size]
    X_test_obs = X_test.iloc[:, :auction_model.obs_embedding_size]

    # for specific experiment settings
    if '100' in args.exp:
        if args.exp == '100_kmeans':
            X_train_obs = shap.kmeans(X_train_obs, int(0.1 * data_size))

        ex_obs = shap.KernelExplainer(simulate_bids, X_train_obs)
        print(ex_obs.expected_value)

        with open('../shap/kernel_' + args.exp + '_' + filename + '.pkl', 'wb') as file:
            pickle.dump(ex_obs, file)
            print(f'Object successfully saved to "kernel_{args.exp}_{filename}.pkl"')

        shap.initjs()

        # Explain all the predictions in the test set
        start_time = time.time()
        shap_values_all_obs = ex_obs.shap_values(X_test_obs)
        print(f'Kernel explainer time of {args.exp}: {timedelta(seconds=time.time() - start_time)}')

        np.savetxt('../shap/kernel_' + args.exp + '_' + filename + '.csv', shap_values_all_obs, delimiter=",")
        shap.summary_plot(shap_values_all_obs, X_test_obs, show=False)
        plt.savefig('../shap/kernel_' + args.exp + '_' + filename + '.pdf')
        print('DONE SUMMARY')

    elif args.exp == 'progress':
        iters = None
        if args.iter == 'begin':
            with open('../model/' + args.way + '/begin_' + filename + '.pkl', 'rb') as file:
                auction_model = pickle.load(file)
                print(f'Object successfully load from "{filename}_begin.pkl"')

            explainer = shap.explainers.Exact(simulate_bids_permutation, X_train_obs)

            start_time = time.time()
            shap_values_permutation = explainer(X_test_obs)
            print(f'Exact explainer time of begin: {timedelta(seconds=time.time() - start_time)}')

            with open('../shap/' + args.way + '/begin_' + filename + '.pkl', 'wb') as file:
                pickle.dump(shap_values_permutation, file)
                print(f'Object successfully saved to "exact_{filename}.pkl"')

            np.savetxt('../shap/' + args.way + '/begin_' + filename + '.csv', shap_values_permutation.values,
                       delimiter=",")
            shap.summary_plot(shap_values_permutation, X_test_obs, show=False)
            plt.savefig('../shap/' + args.way + '/begin_' + filename + '.pdf')
            plt.clf()
            exit(0)
        elif args.iter == 'first':
            iters = [0, 3, 6, 9, 12, 15]
        elif args.iter == 'second':
            iters = [18, 21, 24, 27, 30, 33]
        elif args.iter == 'third':
            iters = [36, 39, 42, 45, 48]
        for i in iters:
            # with open('../shap/' + args.way + '/iter' + str(i) + '_' + filename + '.pkl', 'rb') as file:
            #     shap_values_permutation = pickle.load(file)
            #     print(f'Object successfully load from "exact_{filename}.pkl"')

            with open('../model/' + args.way + '/iter' + str(i) + '_' + filename + '.pkl', 'rb') as file:
                auction_model = pickle.load(file)
                print(f'Object successfully load from "{filename}_iter{str(i)}.pkl"')

            explainer = shap.explainers.Exact(simulate_bids_permutation, X_train_obs)

            start_time = time.time()
            shap_values_permutation = explainer(X_test_obs)
            print(f'Exact explainer time of iter{str(i)}: {timedelta(seconds=time.time() - start_time)}')

            with open('../shap/' + args.way + '/iter' + str(i) + '_' + filename + '.pkl', 'wb') as file:
                pickle.dump(shap_values_permutation, file)
                print(f'Object successfully saved to "exact_{filename}.pkl"')

            np.savetxt('../shap/' + args.way + '/iter' + str(i) + '_' + filename + '.csv', shap_values_permutation.values,
                       delimiter=",")
            shap.summary_plot(shap_values_permutation, X_test_obs, show=False)
            plt.savefig('../shap/' + args.way + '/iter' + str(i) + '_' + filename + '.pdf')
            plt.clf()
        print('DONE SUMMARY')
