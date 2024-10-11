# Sparse Training Environment

---

Code and results of Uncovering the Accuracy-Efficiency Trade-Off in Sparse Neural Network Training. 
This research set multiple goals:
1. To evaluate the validity of previous research, showing how SNN training can vastly improve efficiency without a loss of accuracy.
2. To define a systematic way to research the effects of new technologies with multiple conflicting objectives.
3. To present the accuracy-efficiency trade-off when training arbitrary neural networks with SNN training in a realistic setting.

This is achieved using a set of experiments on which analyses were run later. These experiments are performed on the MNIST, Fashion-MNIST, Higgs, Electricity, CIFAR-10 and SVHN datasets. The first four were modeled using standard multi-layer perceptrons, the last two using residual convolutional networks.

## Experiments
Several types of experiments can be performed, all of which can be defined in YAML config files in /config using Hydra.

#### Reproduction
Uses the default hyperparameter configuration and applies SNN training to it. Evaluates 15 sparsities on SET, 15 on RigL, and trains 1 dense network.

```
python main.py experiment=[dataset] experiment_type=optimization seed=1
```

#### Optimization
Uses MO-SMAC, an MO-HPO optimizer, to find the best configurations. MO-HPO is a stochastic task and should be performed multiple times using different seeds to prevent results getting stuck in local optima.
To perform this first step:

```
python main.py experiment=<dataset> experiment_type=reproduction seed=<seed>
```

Afterward, the results of all seeds should be aggregated. From here, we select the first 2 non-dominated layers and re-evaluate them on the same train and test set with the same seed:

```
python main.py experiment=<dataset> experiment_type=optimization-evaluation seed=<seed>
```

#### Evaluation
Chooses the 20 configurations resulting from the first step of the optimization experiment that yield the highest accuracies. Each of these configurations is re-evaluated as 8 different sparsities on SET, 8 on RigL and one dense network.

```
python main.py experiment=<dataset> experiment_type=evaluation seed=<seed>
```

## Results
Results of all experiments performed on these six datasets with optimization on ten seeds can be found in data/run_dataset.csv.


## Random Search
Next to these specific experiments, you can perform an optimization experiment using a random search with Latin Hypercube Sampling. Such random searches are useful for analyses for which the results need to be unbiased.

```
python main.py experiment=<dataset> experiment_type=random-search seed=<seed>
```

---

## Requirements:
```
conda create -n autosparse python=3.10
conda activate autosparse
conda install gxx_linux-64 gcc_linux-64 swig
pip -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run:
python main.py

Optional arguments:
- experiment=[mnist, fashion_mnist, higgs, electricity, cifar10, svhn]
- experiment.experiment_type=[reproduction, optimization, optimization-evaluation, evaluation, random-search]
- hydra.verbose=[\_\_main\_\_]
- seed=[1-100]
- Any value defined in the configuration files in /config
