# Compare training progress
import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ONLY FOR AUCTION TYPE
# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
# mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors[2:])

social_surplus = {}
auction_revenue = {}
social_welfare = {}
bidder_surplus = {}
bidder_overbid_regret = {}
bidder_underbid_regret = {}

for filename in os.listdir('../training'):
    if filename.endswith('.csv'):
        params = filename.split('_')

        # comparing DM, DR, IPS
        if params[4] == 'default' and params[0].endswith('False') and params[1] == 'FP' and params[3] == '8f':
            print(filename)
            temp = pd.read_csv('../training/' + filename)
            # social_surplus[params[2]] = list(temp['social_surplus'])
            # auction_revenue[params[2]] = list(temp['auction_revenue'])
            # social_welfare[params[2]] = list(temp['social_welfare'])
            bidder_surplus[params[2]] = list(temp['my_surplus'])
            # bidder_overbid_regret[params[2]] = list(temp['my_overbid_regret'])
            # bidder_underbid_regret[params[2]] = list(temp['my_underbid_regret'])

        # # comparing FP, SP
        # if params[4] == 'default' and params[0].endswith('False') and params[2] == 'DR' and params[3] == '8f':
        #     print(filename)
        #     temp = pd.read_csv('../training/' + filename)
        #     social_surplus[params[1]] = list(temp['social_surplus'])
        #     auction_revenue[params[1]] = list(temp['auction_revenue'])
        #     social_welfare[params[1]] = list(temp['social_welfare'])
        #     bidder_surplus[params[1]] = list(temp['my_surplus'])
        #     bidder_overbid_regret[params[1]] = list(temp['my_overbid_regret'])
        #     bidder_underbid_regret[params[1]] = list(temp['my_underbid_regret'])

# for name in ['social_surplus', 'auction_revenue', 'social_welfare', 'bidder_surplus', 'bidder_overbid_regret', 'bidder_underbid_regret']:
for name in ['bidder_surplus']:
    for k in ['DM', 'IPS', 'DR']:
    # for k in ['FP', 'SP']:
        plt.plot(range(1, len(locals()[name][k]) + 1), locals()[name][k], label=k)

    plt.legend()
    plt.ylabel(' '.join([n.capitalize() for n in name.split('_')]))
    plt.xlabel('Iteration')
    # plt.gca().set_ylim(-1, 1)
    # plt.show()
    plt.savefig('../figures/utility_train.pdf')
    # plt.savefig('../figures/auction_train.pdf')
    plt.clf()

