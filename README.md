# From Kernel to Permutation Estimator: Applying SHAP to Explain Reinforcement Learning in Online Advertising
This repository contains the main files for my master's dissertation project (submitted in August 2023) at The University of Edinburgh. The project is industry-supported by Amazon, supervised by Ben Allison, Doudou Tang and Robert Hu, and coordinated by Prof Iain Murray.

## Project Introduction
This project aims to explore how [SHapley Additive exPlanations](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) (**SHAP**), a model-agnostic eXplainable Artificial Intelligence (XAI) method, can explain the behaviours of Reinforcement Learning (**RL**) bidding agents.

### Motivation and Background
[Learning to bid with AuctionGym](https://www.amazon.science/publications/learning-to-bid-with-auctiongym) formulates bidders (advertisers) participating in online advertising auctions as RL agents, whose rewards are utilities that can be estimated. This paper also proposes an auction simulation environment - [AuctionGym](https://github.com/amzn/auction-gym), which supports the use of RL bidders in such auctions. In a simulated auction round, an advertisement opportunity is described by context features. Since real data is not available due to its proprietary and sensitive nature, context features are generated by sampling from data distributions.

While RL has proven to be an effective technique for optimising bidding strategies, its lack of transparency prevents humans from trusting and using it. In this project, we would like to mitigate the problem by employing SHAP. SHAP can fairly assign feature importance/contributions toward a model prediction. Given the computational complexity of precise SHAP value calculations, we focus on two SHAP value estimators - Kernel and Permutation.

### Methodology
We develop a three-step pipeline:
1.  Train a bidder (RL model) in AuctionGym given its configuration file.
2.  Generate data for explanation, with context features following the same data distributions as during training.
3.  Apply SHAP to explain how the bidder behaves on each data instance.

In addition, we also visualise SHAP values through plots and dashboards.

## File Structure
As this project is based on [AuctionGym](https://github.com/amzn/auction-gym), we make modifications to some AuctionGym settings and implement the application of SHAP to bidder models. The project's file structure is organised as follows:

### Modification of AuctionGym
- [src](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/src)/[Auction.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/src/Auction.py): passing ``true_context`` as a parameter of ``simulate_opportunity()``, enabling the use of customised context features for advertisement opportunities.
- [config](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/config): adding configurations for 11 bidders, named as [Auction Type]\_[Utility Estimator]\_[Number of Observable Context Features],
	- Auction Type: **FP** (First-Price), **SP**(Second-Price);
	- Utility Estimator: **DM** (Direct Method), **IPS** (Inverse Propensity Score), **DR** (Doubly Robust);
	- Number of Observable Context Features: **4**/**8**/**16**/**24**/**32**/**40** (**f**eatures).
- [requirements.txt](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/requirements.txt): resolving the NumPy version conflict and adding packages for SHAP.
- [README.md](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/README.md): updating, the original version can be found [here](https://github.com/amzn/auction-gym/blob/main/README.md) .

### Implementation of SHAP Application

 - [pipeline](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/pipeline): containing the implementation of the pipeline described in [Methodology](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning#methodology), including
	 1. [train_bidder.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/pipeline/train_bidder.py): training all bidders with the same configurations simultaneously (adapted from [src](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/src)/[main.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/src/main.py)) but only saving one bidder model for further explanation, and adding multiple data distributions that can be used to sample context features;
	 2. [data_generation.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/pipeline/data_generation.py): sampling new data for explanation, and splitting the data into two sets with a ratio of 0.8 - the background data for SHAP (`X_train`) and the explanation data (`X_test`);
	 3. [SHAP Value Estimator]SHAP&#46;py: estimating SHAP values for each explanation data instance using the corresponding model-agnostic SHAP value estimator supported by the [`shap`](https://github.com/shap/shap) package, as well as saving a summary plot for all SHAP values of the explanation dataset. [SHAP_special.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/pipeline/SHAP_special.py): enabling two special explanation cases - applying k-means when using the Kernel estimator, and explaining several bidder models that are trained with different numbers of iterations using the Permutation estimator.
 - [visualisation](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/visualisation): containing scripts for visualising experimental results, including
	 - [train_plot.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/visualisation/train_plot.py): plotting training-related information;
	 - SHAP_[Experiment]&#46;py: plotting SHAP-related information based on different experiments. An example experiment ([SHAP_utility.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/visualisation/SHAP_utility.py)) is to compare the differences between bidder models that use different utility estimators.
	 - [dashboard_SHAP.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/visualisation/dashboard_SHAP.py): generating web dashboards using the [`shapash`](https://github.com/MAIF/shapash) package.
 - examples: as we save files to several folders within [pipeline](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/pipeline) and [visualisation](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/visualisation), here we provide examples of such files when there are 40 context features, including
	 - [training](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/training): a table with numerical results of several metrics during training and a plot showing four of these metrics over iterations;
	 - [model](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/model): a bidder model saved from the final iteration;
	 - [data](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/data): a background dataset and an explanation one;
	 - [shap](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/shap): a table, an object and a summary plot, all of which contain SHAP values for all explanation data instances;
	 - [dashboard](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/dashboard): a dashboard object that can be called directly.
 - [figure](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/figure): containing all figures created for my dissertation.

## Reproducing Research Results
To replicate the examples presented above, please follow these commands after configuring the environment:

### Pipeline
`cd pipeline`

Train a bidder model with 40 context features that follow a specific data distribution, and save training information to [training](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/training) and the final model to [model](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/model):

`python train_bidder.py FP_DR_40f.json specific True False`

Sample 1000 data instances from the same specific distribution and save them to [data](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/data):

`python data_generation.py FP_DR_40f.json specific True 1000`

Estimate SHAP values for the explanation dataset using Permutation estimator given the bidder model, and save the value information to [shap](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/shap):

`python permutationSHAP.py FP_DR_40f.json specific True 1000`  

### Dashboard
`cd visualisation`

Display the dashboard saved in [dashboard](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/tree/main/dashboard) on a browser (can also re-create one by commenting and uncommenting codes in [dashboard_SHAP.py](https://github.com/jingxuanchen916/Applying-SHAP-to-Explain-Reinforcement-Learning/blob/main/visualisation/dashboard_SHAP.py)):

`python dashboard_SHAP.py FP_DR_40f.json specific True 1000`
