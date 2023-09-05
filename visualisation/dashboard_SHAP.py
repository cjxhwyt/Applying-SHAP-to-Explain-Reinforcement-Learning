# Use shapash to create web dashboard
import sys

sys.path.insert(1, '../src')

import argparse
import numpy as np
import pandas as pd
import pickle
import shapash
import torch
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


class Simulation:
    def __call__(self, x):
        self.predict(x)

    def predict(self, all_contexts):
        result = []

        for _, contexts in all_contexts.iterrows():
            result.append(simulate_a_bid(contexts))

        return np.array(result)


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('way', type=str, help='Way to generate data')
    parser.add_argument('specific', type=str, help='Whether to generate specific data')
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

    model = Simulation()

    y_pred = pd.DataFrame(
        model.predict(X_test_obs), columns=['pred'], index=X_test_obs.index
    )

    # create dashboard and save
    # xpl = shapash.SmartExplainer(model=model, explainer_args={'masker': X_train_obs, 'algorithm': 'permutation'})
    #
    # start_time = time.time()
    # xpl.compile(y_pred=y_pred, x=X_test_obs)
    # print(f'Dashboard time: {timedelta(seconds=time.time() - start_time)}')
    #
    # xpl.save('../dashboard/db_' + filename + '.pkl')
    # print(f'Object successfully saved to "dashboard/{filename}.pkl"')

    # read existing dashboard
    with open('../dashboard/db_' + filename + '.pkl', 'rb') as file:
        xpl = pickle.load(file)
        print(f'Object successfully load from "dashboard/{filename}.pkl"')

    app = xpl.run_app(title_story='XRL with AuctionGym')
