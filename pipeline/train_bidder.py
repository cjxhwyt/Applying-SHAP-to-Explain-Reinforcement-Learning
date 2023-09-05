# Step1: train a bidder (RL model) in AuctionGym given its configuration file
import sys

sys.path.insert(1, '../src')

import matplotlib.pyplot as plt
import numpy as np
from main import parse_config, instantiate_agents, instantiate_auction
from tqdm.notebook import tqdm

import pickle
import pandas as pd
import argparse

import time
from datetime import timedelta


# MODIFICATION: add an agent id to represent 'myself'
def run_repeated_auctions(config_filename, way, filename):
    # Parse configuration file
    rng, config, agent_configs, agents2items, agents2item_values, num_runs, max_slots, embedding_size, embedding_var, \
        obs_embedding_size = parse_config('../config/' + config_filename)

    print("AGENTS2ITEMS")
    print(agents2items)
    print(agents2item_values)

    # Placeholders for output
    auction_revenue = []
    social_welfare = []
    social_surplus = []
    # MODIFICATION: add another output for 'my' information
    my_surplus = []
    my_welfare = []
    my_allocation_regret = []
    my_estimation_regret = []
    my_overbid_regret = []
    my_underbid_regret = []
    my_CTR_RMSE = []
    my_CTR_bias = []
    my_best_expected_value = []

    # Instantiate Agent and Auction objects
    agents = instantiate_agents(rng, agent_configs, agents2item_values, agents2items)

    # Instantiate Auction object
    auction, num_iter, rounds_per_iter, output_dir = \
        instantiate_auction(rng,
                            config,
                            agents2items,
                            agents2item_values,
                            agents,
                            max_slots,
                            embedding_size,
                            embedding_var,
                            obs_embedding_size)

    # Run repeated auctions
    # This logic is encoded in the `simulation_run()` method in main.py
    for i in tqdm(range(num_iter)):

        # save checkpoint when required
        if progress:
            if i % 3 == 0:
                with open('../model/' + way + '/iter' + str(i) + '_' + filename + '.pkl', 'wb') as f:
                    pickle.dump(auction, f)
                    print(f'Object successfully saved to "{filename}_iter{str(i)}.pkl"')

        # Simulate impression opportunities
        for _ in range(rounds_per_iter):
            if not specific:
                # default setting of data
                contexts = np.concatenate((auction.rng.normal(0, auction.embedding_var, size=auction.embedding_size)
                                           , [1.0]))
                weights = np.concatenate(([-4, -2, -1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
                # different data distributions for 8 observable features
                if auction.embedding_size - 2 == 8:
                    if way == 'noise':
                        weights = np.concatenate(([1 for _ in range(auction.embedding_size)], [1.0]))
                    elif way == 'absolute':
                        weights = np.concatenate(([4, 2, 1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
                    elif way == 'random':
                        weights = np.concatenate(([1, 4, -2, 8, -4, -1, 0, 2, 1, 1], [1.0]))
                    elif way == 'zero':
                        weights = np.concatenate(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1.0]))
                    elif way == 'two':
                        weights = np.concatenate(([2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1.0]))
                    elif way == 'five':
                        weights = np.concatenate(([1, 1, 1, 5, 1, 1, 1, 1, 1, 1], [1.0]))
                    elif way == 'ten':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 10, 1, 1, 1, 1], [1.0]))
                    elif way == 'two_copy':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 2, 1, 1], [1.0]))
                    elif way == 'five_copy':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 5, 1, 1], [1.0]))
                    elif way == 'ten_copy':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 10, 1, 1], [1.0]))
                    elif way == 'fifteen':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 15, 1, 1], [1.0]))
                    elif way == 'two_negative':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -2, 1, 1], [1.0]))
                    elif way == 'five_negative':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -5, 1, 1], [1.0]))
                    elif way == 'ten_negative':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -10, 1, 1], [1.0]))
                    elif way == 'fifteen_negative':
                        weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -15, 1, 1], [1.0]))
                    elif way == 'uniform':
                        contexts = np.concatenate((auction.rng.uniform(0, 1, size=auction.embedding_size), [1.0]))
                    elif way == 'uniform_positive':
                        contexts = np.concatenate((auction.rng.uniform(0, 1, size=auction.embedding_size), [1.0]))
                        weights = np.concatenate(([1 for _ in range(auction.embedding_size)], [1.0]))
                    elif way == 'uniform_negative':
                        contexts = np.concatenate((auction.rng.uniform(0, 1, size=auction.embedding_size), [1.0]))
                        weights = np.concatenate(([-1 for _ in range(auction.embedding_size)], [1.0]))
                    elif way == 'uniform_opposite':
                        contexts = np.concatenate((auction.rng.uniform(0, 1, size=auction.embedding_size), [1.0]))
                        weights = -weights
                    elif way == 'uniform_one':
                        contexts = np.concatenate((auction.rng.uniform(-1, 1, size=auction.embedding_size), [1.0]))
                    elif way == 'uniform_two':
                        contexts = np.concatenate((auction.rng.uniform(-2, 2, size=auction.embedding_size), [1.0]))
                    elif way == 'opposite':
                        weights = -weights
                    elif way == 'opposite_one':
                        weights = np.concatenate(([4, -2, -1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
                    elif way == 'variance':
                        contexts_obs = [auction.rng.normal(0, auction.embedding_var * w) for w in
                                        range(1, auction.embedding_size - 1)]
                        contexts = np.hstack((contexts_obs, [1, 1, 1]))
                        weights = np.array([1 for _ in range(auction.embedding_size + 1)])
                    elif way == 'variance_opposite':
                        contexts_obs = [auction.rng.normal(0, auction.embedding_var * w) for w in
                                        range(auction.embedding_size - 2, 0, -1)]
                        contexts = np.hstack((contexts_obs, [1, 1, 1]))
                        weights = np.array([1 for _ in range(auction.embedding_size + 1)])
                    elif way == 'variance_replaced':
                        contexts_obs = [auction.rng.normal(0, auction.embedding_var * w) for w in
                                        range(1, auction.embedding_size - 2)]
                        contexts_obs_2 = auction.rng.normal(0, auction.embedding_var)
                        contexts = np.hstack((contexts_obs, contexts_obs_2, [1, 1, 1]))
                        weights = np.array([1 for _ in range(auction.embedding_size + 1)])
                # weights for 4 features
                elif auction.embedding_size - 1 == 4:
                    weights = np.concatenate(([2, 0, -4, 8, 1], [1.0]))
                    if way == 'noise':
                        weights = np.array([1 for _ in range(auction.embedding_size + 1)])
                else:
                    weights = np.array([1 for _ in range(auction.embedding_size + 1)])
                contexts *= weights
            # specific data distributions when the number of observable features is 8/16/24/32/40/...
            else:
                q = auction.embedding_size / 10  # use to loop for specific feature num
                contexts = []
                if q.is_integer():
                    for _ in range(int(q)):
                        # 3
                        context_normal_1 = auction.rng.normal(0, auction.embedding_var)
                        context_normal_2 = auction.rng.normal(0, 5 * auction.embedding_var)
                        context_normal_3 = auction.rng.normal(5, auction.embedding_var)
                        # 3
                        context_uniform = auction.rng.uniform(low=-2, high=2, size=3)
                        context_uniform *= np.array([-5, 0, 5])
                        # 1
                        context_binary = auction.rng.integers(0, 2)
                        # 1
                        context_category = auction.rng.integers(1, 6)
                        # combine
                        contexts = np.hstack((contexts, context_normal_1, context_normal_2, context_normal_3,
                                             context_uniform, context_binary, context_category))
                    contexts = np.hstack((contexts, [1 for _ in range(int(auction.embedding_size - q * 8))], [1.0]))
            assert len(contexts) == auction.embedding_size + 1
            # run an auction opportunity
            auction.simulate_opportunity(contexts)

        # Log 'Gross utility' or welfare
        social_welfare.append(sum([agent.gross_utility for agent in auction.agents]))

        # Log 'Net utility' or surplus
        social_surplus.append(sum([agent.net_utility for agent in auction.agents]))

        # MODIFICATION
        # Log 'my' information
        my_surplus.append(auction.agents[my_agent_id].net_utility)
        my_welfare.append(auction.agents[my_agent_id].gross_utility)

        my_allocation_regret.append(auction.agents[my_agent_id].get_allocation_regret())
        my_estimation_regret.append(auction.agents[my_agent_id].get_estimation_regret())
        my_overbid_regret.append(auction.agents[my_agent_id].get_overbid_regret())
        my_underbid_regret.append(auction.agents[my_agent_id].get_underbid_regret())

        my_CTR_RMSE.append(auction.agents[my_agent_id].get_CTR_RMSE())
        my_CTR_bias.append(auction.agents[my_agent_id].get_CTR_bias())

        best_expected_value = np.mean([opp.best_expected_value for opp in auction.agents[my_agent_id].logs])
        my_best_expected_value.append(best_expected_value)

        # Update agents
        # Clear running metrics
        for agent_id, agent in enumerate(auction.agents):
            agent.update(iteration=i)
            agent.clear_utility()
            agent.clear_logs()

        # Log revenue
        auction_revenue.append(auction.revenue)
        auction.clear_revenue()

    # Rescale metrics per auction round
    auction_revenue = np.array(auction_revenue) / rounds_per_iter
    social_welfare = np.array(social_welfare) / rounds_per_iter
    social_surplus = np.array(social_surplus) / rounds_per_iter
    # MODIFICATION: rescale 'my' information
    my_surplus = np.array(my_surplus) / rounds_per_iter
    my_welfare = np.array(my_welfare) / rounds_per_iter
    my_allocation_regret = np.array(my_allocation_regret) / rounds_per_iter
    my_estimation_regret = np.array(my_estimation_regret) / rounds_per_iter
    my_overbid_regret = np.array(my_overbid_regret) / rounds_per_iter
    my_underbid_regret = np.array(my_underbid_regret) / rounds_per_iter
    my_CTR_RMSE = np.array(my_CTR_RMSE) / rounds_per_iter
    my_CTR_bias = np.array(my_CTR_bias) / rounds_per_iter
    my_best_expected_value = np.array(my_best_expected_value) / rounds_per_iter

    # save information
    train_info = pd.DataFrame(np.vstack((auction_revenue, social_welfare, social_surplus, my_surplus, my_welfare,
                                         my_allocation_regret, my_estimation_regret, my_overbid_regret,
                                         my_underbid_regret, my_CTR_RMSE, my_CTR_bias, my_best_expected_value)).T,
                              columns=['auction_revenue', 'social_welfare', 'social_surplus', 'my_surplus',
                                       'my_welfare', 'my_allocation_regret', 'my_estimation_regret',
                                       'my_overbid_regret', 'my_underbid_regret', 'my_CTR_RMSE',
                                       'my_CTR_bias', 'my_best_expected_value'])
    train_info.to_csv('../training/progress' + str(progress) + '_' + filename + '_train_info.csv', index=False)

    # save figure
    fontsize = 16
    fig, axes = plt.subplots(1, 4, sharey='row', figsize=(15, 4))
    axes[0].plot(social_welfare)
    axes[1].plot(social_surplus)
    axes[2].plot(auction_revenue)
    axes[3].plot(my_surplus)
    for i in range(4):
        axes[i].set_xlabel('Iterations', fontsize=fontsize)
        axes[i].set_xticks(list(range(0, len(my_surplus), 2)))
        axes[i].grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
    axes[0].set_ylabel('Social Welfare', fontsize=fontsize)
    axes[1].set_ylabel('Social Surplus', fontsize=fontsize)
    axes[2].set_ylabel('Auction Revenue', fontsize=fontsize)
    axes[3].set_ylabel('My Surplus', fontsize=fontsize)
    fig.tight_layout()
    plt.savefig('../training/progress' + str(progress) + '_' + filename + '_train_info.pdf')

    # MODIFICATION: add more return variables
    return auction


if __name__ == '__main__':
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to experiment configuration file')
    parser.add_argument('way', type=str, help='Way to generate data')
    parser.add_argument('specific', type=str, help='Whether to generate data with specific distribution')
    parser.add_argument('progress', type=str, help='Whether to save models during training progress')
    args = parser.parse_args()
    # change to bool type
    specific = eval(args.specific)
    progress = eval(args.progress)

    filename = args.config[:-5] + '_' + args.way + '_' + str(specific)

    my_agent_id = 0
    start_time = time.time()
    auction_model = run_repeated_auctions(args.config, args.way, filename)
    print(f'Training time: {timedelta(seconds=time.time() - start_time)}')

    # save model
    if not progress:
        with open('../model/' + filename + '.pkl', 'wb') as file:
            pickle.dump(auction_model, file)
            print(f'Object successfully saved to "{filename}.pkl"')
