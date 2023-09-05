# Step3: apply SHAP (Kernel Estimator) to explain how the bidder behaves on each data instance
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
    parser.add_argument('specific', type=str, help='Whether to generate data with specific distribution')
    parser.add_argument('size', type=int, help='Size of data')
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

    ex_obs = shap.KernelExplainer(simulate_bids, X_train_obs)
    print(ex_obs.expected_value)

    with open('../shap/kernel_' + filename + '.pkl', 'wb') as file:
        pickle.dump(ex_obs, file)
        print(f'Object successfully saved to "kernel_{filename}.pkl"')

    shap.initjs()

    # Explain all the predictions in the test set
    start_time = time.time()
    shap_values_all_obs = ex_obs.shap_values(X_test_obs)
    print(f'Kernel explainer time: {timedelta(seconds=time.time() - start_time)}')

    np.savetxt('../shap/kernel_' + filename + '.csv', shap_values_all_obs, delimiter=",")
    shap.summary_plot(shap_values_all_obs, X_test_obs, show=False)
    plt.savefig('../shap/kernel_' + filename + '.pdf')
    print('DONE SUMMARY')
