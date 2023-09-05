# Step2: generate data for explanation, with context features following the same data distributions as during training
import sys

sys.path.insert(1, '../src')

import argparse
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

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

    filename = args.config[:-5] + '_' + args.way + '_' + str(specific)
    with open('../model/' + filename + '.pkl', 'rb') as file:
        auction_model = pickle.load(file)
        print(f'Object successfully load from "{filename}.pkl"')

    features = ['Feature ' + str(i + 1) for i in range(auction_model.embedding_size)]
    features.append('placeholder')
    # same data distribution logic as train_bidder.py, but generate several instances in one go
    one_contexts = np.array([[1.0] for _ in range(data_size)])
    if not specific:
        other_contexts = auction_model.rng.normal(0, auction_model.embedding_var,
                                                  size=(data_size, auction_model.embedding_size))
        weights = np.concatenate(([-4, -2, -1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
        if auction_model.embedding_size - 2 == 8:
            if args.way == 'noise':
                weights = np.concatenate(([1 for _ in range(auction_model.embedding_size)], [1.0]))
            elif args.way == 'absolute':
                weights = np.concatenate(([4, 2, 1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
            elif args.way == 'random':
                weights = np.concatenate(([1, 4, -2, 8, -4, -1, 0, 2, 1, 1], [1.0]))
            elif args.way == 'zero':
                weights = np.concatenate(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1.0]))
            elif args.way == 'two':
                weights = np.concatenate(([2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1.0]))
            elif args.way == 'five':
                weights = np.concatenate(([1, 1, 1, 5, 1, 1, 1, 1, 1, 1], [1.0]))
            elif args.way == 'ten':
                weights = np.concatenate(([1, 1, 1, 1, 1, 10, 1, 1, 1, 1], [1.0]))
            elif args.way == 'two_copy':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 2, 1, 1], [1.0]))
            elif args.way == 'five_copy':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 5, 1, 1], [1.0]))
            elif args.way == 'ten_copy':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 10, 1, 1], [1.0]))
            elif args.way == 'fifteen':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, 15, 1, 1], [1.0]))
            elif args.way == 'two_negative':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -2, 1, 1], [1.0]))
            elif args.way == 'five_negative':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -5, 1, 1], [1.0]))
            elif args.way == 'ten_negative':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -10, 1, 1], [1.0]))
            elif args.way == 'fifteen_negative':
                weights = np.concatenate(([1, 1, 1, 1, 1, 1, 1, -15, 1, 1], [1.0]))
            elif args.way == 'uniform':
                other_contexts = auction_model.rng.uniform(0, 1, size=(data_size, auction_model.embedding_size))
            elif args.way == 'uniform_positive':
                other_contexts = auction_model.rng.uniform(0, 1, size=(data_size, auction_model.embedding_size))
                weights = np.concatenate(([1 for _ in range(auction_model.embedding_size)], [1.0]))
            elif args.way == 'uniform_negative':
                other_contexts = auction_model.rng.uniform(0, 1, size=(data_size, auction_model.embedding_size))
                weights = np.concatenate(([-1 for _ in range(auction_model.embedding_size)], [1.0]))
            elif args.way == 'uniform_opposite':
                other_contexts = auction_model.rng.uniform(0, 1, size=(data_size, auction_model.embedding_size))
                weights = -weights
            elif args.way == 'uniform_one':
                other_contexts = auction_model.rng.uniform(-1, 1, size=(data_size, auction_model.embedding_size))
            elif args.way == 'uniform_two':
                other_contexts = auction_model.rng.uniform(-2, 2, size=(data_size, auction_model.embedding_size))
            elif args.way == 'opposite':
                weights = -weights
            elif args.way == 'opposite_one':
                weights = np.concatenate(([4, -2, -1, 0, 1, 2, 4, 8, 1, 1], [1.0]))
            elif args.way == 'variance':
                contexts_obs = np.array([auction_model.rng.normal(0, auction_model.embedding_var * w, size=data_size)
                                         for w in range(1, auction_model.embedding_size - 1)]).T
                other_contexts = np.hstack((contexts_obs, one_contexts, one_contexts))
                weights = np.concatenate(([1 for _ in range(auction_model.embedding_size)], [1.0]))
            elif args.way == 'variance_opposite':
                contexts_obs = np.array([auction_model.rng.normal(0, auction_model.embedding_var * w, size=data_size)
                                         for w in range(auction_model.embedding_size - 2, 0, -1)]).T
                other_contexts = np.hstack((contexts_obs, one_contexts, one_contexts))
                weights = np.concatenate(([1 for _ in range(auction_model.embedding_size)], [1.0]))
            elif args.way == 'variance_replaced':
                contexts_obs = np.array([auction_model.rng.normal(0, auction_model.embedding_var * w, size=data_size)
                                         for w in range(1, auction_model.embedding_size - 2)]).T
                contexts_obs_2 = auction_model.rng.normal(0, auction_model.embedding_var, size=(data_size, 1))
                other_contexts = np.hstack((contexts_obs, contexts_obs_2, one_contexts, one_contexts))
                weights = np.concatenate(([1 for _ in range(auction_model.embedding_size)], [1.0]))
        elif auction_model.embedding_size - 1 == 4:
            weights = np.concatenate(([2, 0, -4, 8, 1], [1.0]))
        else:
            weights = np.array([1 for _ in range(auction_model.embedding_size + 1)])
        data = weights * np.hstack((other_contexts, one_contexts))
    else:
        q = auction_model.embedding_size / 10  # use to loop for specific feature num
        if q.is_integer():
            for j in range(int(q)):
                # 3
                context_normal_1 = auction_model.rng.normal(0, auction_model.embedding_var, size=(data_size, 1))
                context_normal_2 = auction_model.rng.normal(0, 5 * auction_model.embedding_var, size=(data_size, 1))
                context_normal_3 = auction_model.rng.normal(5, auction_model.embedding_var, size=(data_size, 1))
                # 3
                context_uniform = auction_model.rng.uniform(low=-2, high=2, size=(data_size, 3))
                context_uniform *= np.array([-5, 0, 5])
                # 1
                context_binary = auction_model.rng.integers(0, 2, size=(data_size, 1))
                # 1
                context_category = auction_model.rng.integers(1, 6, size=(data_size, 1))
                # combine
                if j == 0:
                    data = np.hstack((context_normal_1, context_normal_2, context_normal_3, context_uniform,
                                      context_binary, context_category))
                else:
                    data = np.hstack((data, context_normal_1, context_normal_2, context_normal_3, context_uniform,
                                      context_binary, context_category))
            data = np.hstack((data, np.hstack([one_contexts for _ in range(int(auction_model.embedding_size - q * 8))]),
                              one_contexts))
    assert data.shape == (data_size, auction_model.embedding_size + 1)

    X_train, X_test = train_test_split(data, test_size=0.2, random_state=0)
    X_train = pd.DataFrame(X_train, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)

    X_train.to_csv('../data/' + filename + '_' + str(data_size) + '_X_train.csv', index=False)
    X_test.to_csv('../data/' + filename + '_' + str(data_size) + '_X_test.csv', index=False)

    print('Data generation done!')
