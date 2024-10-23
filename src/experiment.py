import time
import logging
import json
import math
import pickle
import os
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, GreaterThanCondition, NotEqualsCondition, EqualsCondition, Float, Integer, Categorical, Constant
from ConfigSpace.read_and_write import json as cs_json
from omegaconf import DictConfig
from smac import Scenario
from smac.initial_design.latin_hypercube_design import LatinHypercubeInitialDesign
from enum import Enum

from mo_smac import MOFacade
from my_datasets import Dataset
from my_networks import NetworkArchitecture
from sparse_optimization import SparseOptimization
from dataset_loader import load_dataset
from runhistory import RunHistory

logger = logging.getLogger(__name__)


class HyperparameterType(Enum):
    """
    Enum for the different types of hyperparameters.
    """
    REAL = 'r'
    INTEGER = 'i'
    CATEGORICAL = 'c'
    CONSTANT = 'constant'


class Experiment:
    """
    Experiment helper class for evaluating configurations using SparseOptimization.
    An Experiment instance always uses the same pre-loaded dataset and architecture for all evaluations.
    Can be used as "with Experiment() as experiment:".
    If a path is given, automatically builds a json list with all necessary information of all configurations.
    """
    dataset: Dataset = None
    configspace: ConfigurationSpace = None
    architecture: str = None
    budget: int = None
    num_instances: int = None
    seed: int = None
    path: str | None = None
    config_counter: int = None
    time_since_previous_config: float = None
    
    
    def __init__(self, cfg: DictConfig, seed: int = 43, path: str = None):
        self.architecture = cfg.architecture
        self.num_instances = 5
        self.seed = seed
        self.path = path
        self.config_counter = 0
        self.time_since_previous_config = time.time()
        
        logger.info(f"Initializing experiment with seed {seed} at path {path}")
        
        if 'budget' in cfg.keys():
            self.budget = cfg.budget
        if 'hyperparameters' in cfg.keys():
            self.configspace = self.create_config_space(cfg.hyperparameters)
            logger.info(f"Configspace created")
            logger.info(repr(self.configspace))
        
        self.dataset = load_dataset(cfg.dataset, self.num_instances, seed)
        
    
    def __enter__(self):
        """
        Allows with Experiment() as experiment: syntax.
        """
        if self.path:
            cs_path = self.path + "/configspace.json"
            logger.info(f"Storing configurationspace: {cs_path}")
            os.makedirs(os.path.dirname(cs_path), exist_ok=True)
            with open(cs_path, "w+") as file:
                file.write(cs_json.write(self.configspace))
                
            self.path += "/objectives_complete.json"
            logger.info(f"Creating objectives json: {self.path}")
            with open(self.path, "w") as file:
                json.dump({"results": []}, file, indent=4)
        return self
    
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Called if the with block exits, either normally or with an exception.
        Writes the results to a json file if a path was given.
        """
        if self.path:
            logger.info(f"Finished optimization, results written to json: {self.path}")
            with open(self.path, "r") as file:
                res = json.load(file)
         
    
    def store(self, content):
        """
        If the output needs to be stored, append it to the file.
        """
        if self.path:
            logger.info(f"Adding config to json: {self.path}")
            with open(self.path, "r") as file:
                res = json.load(file)
            res['results'].append(content)
            with open(self.path, "w") as file:
                json.dump(res, file, indent=4)
    

    def create_config_space(self, hyperparameters: DictConfig) -> ConfigurationSpace:
        """
        Create the ConfigSpace for the optimization experiment given the Hydra hyperparameters.
        Skips disabled hyperparameters and sets conditions for size mlp_middle and mlp_last layers.
        """
        configspace = {}
        for key, value in hyperparameters.items():
            if not value: # Skip disabled (set to null) hyperparameters
                continue
            if value.type == HyperparameterType.REAL.value:
                configspace[key] = Float(
                    key,
                    bounds=(value.min, value.max),
                    log=value.scaling=='log',
                    default=value.default
                    )
            elif value.type == HyperparameterType.INTEGER.value:
                configspace[key] = Integer(
                    key,
                    bounds=(value.min, value.max),
                    log=value.scaling=='log',
                    default=value.default
                    )
            elif value.type == HyperparameterType.CATEGORICAL.value:
                configspace[key] = Categorical(
                    key,
                    value.options,
                    default=value.default,
                    ordered=value.ordered
                    )
            elif value.type == HyperparameterType.CONSTANT.value:
                # Configspace can't deal with constant boolean values. So we convert them to strings only to convert them back later in sparse optimization.
                if type(value.default) == bool:
                    configspace[key] = Constant(key, str(value.default))
                else:
                    configspace[key] = Constant(key, value.default)
        
        # Manual conditions for use_* hyperparameters
        if hyperparameters['use_dropout'].type == HyperparameterType.CONSTANT.value and hyperparameters['use_dropout'].default == False:
            configspace['dropout'] = Constant('dropout', 0)
        if hyperparameters['use_label_smoothing'].type == HyperparameterType.CONSTANT.value and hyperparameters['use_label_smoothing'].default == False:
            configspace['label_smoothing'] = Constant('label_smoothing', 0)
        
        # Convert the dictionary to a ConfigurationSpace
        configspace = ConfigurationSpace(configspace)
        
        # Add conditions
        if type(configspace['algorithm']) != Constant:
            configspace.add_condition(NotEqualsCondition(configspace['sparsity'], configspace['algorithm'], 'no_prune'))
            configspace.add_condition(NotEqualsCondition(configspace['sparsity_distribution'], configspace['algorithm'], 'no_prune'))
            configspace.add_condition(NotEqualsCondition(configspace['update_freq'], configspace['algorithm'], 'no_prune'))
            configspace.add_condition(NotEqualsCondition(configspace['update_end'], configspace['algorithm'], 'no_prune'))
        if "num_mlp_layers" in configspace.keys() and type(configspace['num_mlp_layers']) != Constant:
            configspace.add_condition(GreaterThanCondition(configspace['size_last_mlp_layer'], configspace['num_mlp_layers'], 1))
            configspace.add_condition(GreaterThanCondition(configspace['size_middle_mlp_layer'], configspace['num_mlp_layers'], 2))
        if type(configspace['use_dropout']) != Constant:
            configspace.add_condition(EqualsCondition(configspace['dropout'], configspace['use_dropout'], True))
        if type(configspace['use_label_smoothing']) != Constant:
            configspace.add_condition(EqualsCondition(configspace['label_smoothing'], configspace['use_label_smoothing'], True))
            
        return configspace


    def train_model(self, sparse_optimizer: SparseOptimization, epochs: int) -> float:
        """
        Trains a SparseOptimization model for a given number of epochs.
        """
        for epoch in range(epochs):
            epoch_start_time = time.time()
            train_loss, train_accuracy = sparse_optimizer.train_model()
            logger.info(f"Epoch: {epoch + 1}, Train loss: {train_loss}, Train accuracy: {train_accuracy}, Time: {time.time() - epoch_start_time}")
            
            if (epoch + 1) % 5 == 0:
                epoch_start_time = time.time()
                val_loss, val_accuracy = sparse_optimizer.validate_model()
                logger.info(f"Epoch: {epoch + 1}, Validation loss: {val_loss}, Validation accuracy: {val_accuracy}, Time: {time.time() - epoch_start_time}")
            
        _, val_accuracy = sparse_optimizer.validate_model()
            
        return float(val_accuracy)


    def evaluate_configuration(self, config: Configuration | dict, seed: int = 0, instance: str = "0"):
        """
        Evaluate a configuration using the SparseOptimization object.
        Handles logging and writing the results to a json file.
        Each epoch, a model is trained and evaluated.
        """
        # Cast types
        config = dict(config)
        instance = int(instance)
        
        # The seed argument is needed for optimization with SMAC, but we use the same seed for all evaluations in an experiment.
        seed = self.seed
        
        logger.info(f"Evaluating configuration {self.config_counter + 1}. Time since previous config: {time.time() - self.time_since_previous_config}. Configuration: {json.dumps(config, indent=4)}")
        logger.info(f'With configuration seed {seed} and database fold {instance}')
        
        # Update config info
        self.time_since_previous_config = time.time()
        self.config_counter += 1
        
        if 'config_id' not in config.keys():
            config['config_id'] = f"{seed}-{self.config_counter}"
        
        self.dataset.prepare_dataset(instance, config["batch_size"])
        
        sparse_optimizer = SparseOptimization(
            dataset=self.dataset,
            architecture=self.architecture,
            prng_key=seed,
            **config
        )
        logger.info(f"Model initialized.")
        
        inference_flops, training_flops = sparse_optimizer.get_model_efficiency()
        logger.info(f"Inference flops: {inference_flops}, Training flops: {training_flops}")
        
        validation_accuracy = self.train_model(sparse_optimizer, config["epochs"])
        error_percentage = 1 - validation_accuracy
        
        self.store({
            'config': config,
            'seed': seed,
            'instance': instance,
            'error_percentage': error_percentage,
            'inference_flops': inference_flops,
            'training_flops': training_flops,
        })
        
        logger.info(json.dumps({"accuracy": validation_accuracy, "error_percentage": error_percentage, "inference_efficiency": inference_flops, "training_efficiency": training_flops}, indent=4))
        
        return {"error_percentage": error_percentage, "inference_efficiency": inference_flops, "training_efficiency": training_flops}


    def reproduction_experiment(self) -> None:
        """
        Perform a reproduction experiment (with fixed configs and seeds etc) for this experiment.
        Returns the evaluated configurations and their objectives, can be ignored as the experiment logs its own results.
        Evaluates 17 * 2 + 1 = 35 configurations, showing the possible range of efficiency-accuracy inferred by sparsity within the default configurations.
        """
        # [round(x, 2) for x in np.linspace(0.01, 0.99, 17)]
        # [0.01, 0.07, 0.13, 0.19, 0.26, 0.32, 0.38, 0.44, 0.5, 0.56, 0.62, 0.68, 0.74, 0.81, 0.87, 0.93, 0.99]
        # Manual sparsities better (totaal 15 ipv 17): 
        # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
        
        logger.info("Defining configurations to evaluate")
        config = dict(self.configspace.get_default_configuration())
        # Configs will be filled with copies of the default config, with different algorithms and sparsities
        configs = []
        config['algorithm'] = 'no_prune'
        config['sparsity'] = 0.0
        configs.append(config.copy())
        
        for algorithm in ['set', 'rigl']:
            config['algorithm'] = algorithm
            for sparsity in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]:
                config['sparsity'] = sparsity
                configs.append(config.copy())
                
        logger.info("Evaluating configurations")
        results = []    
        for config in configs:
            # Instance -1 denotes the entire training set and validation on the test set.
            # Seed 43 to have the same seed as the incumbents
            objs = self.evaluate_configuration(config, seed=43, instance="-1")
            results.append((config, objs))
        return results
    
    
    def optimization_experiment(
        self,
        output_dir: str,
        scenario_name: str,
        seed: int,
        n_workers: int,
        use_seeds: bool,
        use_instances: bool
        ) -> None:
        """
        Initialize and start the SMAC optimization using the given scenario and experiment.
        """
        logger.info("Created configuration space")
        
        scenario = Scenario(
            configspace=self.configspace,
            n_trials=self.budget,
            objectives=self.get_objectives(),
            crash_cost=self.get_crash_costs(),
            output_directory=output_dir,
            name=scenario_name,
            seed=seed,
            n_workers=n_workers,
            deterministic=not use_seeds,
            instances=[str(i) for i in range(self.num_instances)] if use_instances else ["0"],
            instance_features={str(i): [float(i)] for i in range(self.num_instances)} if use_instances else {"0": [0.0]},
            )
        smac = MOFacade(scenario, self.evaluate_configuration)
        incumbents = smac.optimize()
        
        logger.info(f"Incumbent: {incumbents}")

        with open(f"{output_dir}/{scenario_name}/{seed}/runhistory.obj", "wb") as file:
            # Dump the smac runhistory object to a file
            pickle.dump(smac.runhistory, file)
            logger.info("Smac runhistory stored")
            file.close()

    
    def optimization_experiment_evaluation(self, optimization_path: str, seed: int = 43) -> None:
        """
        Evaluate the first 2 non-dominated layers of the optimization runhistory on the entire train and test set.
        """
        runhistory = RunHistory.from_json_complete(optimization_path)
        incumbents = runhistory.get_non_dominated_layers(2)
        incumbents = incumbents[0] + incumbents[1]
        logger.info(f"Evaluating {len(incumbents)} incumbents")
        for incumbent in incumbents:
            # Instance -1 denotes the entire training set and validation on the test set.
            self.evaluate_configuration(incumbent, seed=seed, instance="-1")
    
    
    def evaluation_experiment(self, optimization_path: str, amount: int = 20, seed: int = 43) -> None:
        """
        Choose amount of incumbents from the runhistory and evaluate them with different algorithms and sparsities.
        Evaluates amount * (1 + 2 * 8) = amount * 17 configurations. Default 340
        """
        runhistory = RunHistory.from_json_complete(optimization_path)
        incumbents = runhistory.get_pareto_set()
        sorted_incumbents = list(sorted(incumbents, key=lambda config: runhistory.average_objectives(config)[0]))
        logger.info(f"Evaluating {amount} incumbents")
        for i, config in enumerate(sorted_incumbents[:amount]):
            logger.info(f"Evaluating incumbent nr {i}, {config}")
            config['algorithm'] = 'no_prune'
            config['sparsity'] = 0.0
            self.evaluate_configuration(config.copy(), seed=seed, instance="-1")
            for algorithm in ['set', 'rigl']:
                config['algorithm'] = algorithm
                for sparsity in [0.2, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
                    config['sparsity'] = sparsity
                    self.evaluate_configuration(config.copy(), seed=seed, instance="-1")


    def random_search_experiment(self):
        """
        Evaluate <budget> random configurations sampled from the configspace.
        """
        logger.info("Starting random search")
        configs = self.configspace.sample_configuration(self.budget)
        logger.info(f"Created {len(configs)} random configurations.")
        for config in configs:
            self.evaluate_configuration(config)
        logger.info("Random search finished")


    def latin_random_search_experiment(self, seed: int):
        """
        Evaluate <budget> configurations sampled from the configspace with Latin Hypercube Sampling.
        """
        logger.info("Starting random search with latin hypercube sampling")
        scenario = Scenario(
            configspace=self.configspace,
            n_trials=self.budget,
            objectives=self.get_objectives(),
            crash_cost=self.get_crash_costs(),
            )
        
        # LHS can't sample the exact amount of configurations, so we sample more using several seeds and select the first <budget> configurations
        logger.info("Selecting configurations")
        configs = []
        i = 0
        while len(configs) < self.budget:
            i += 1
            lhd = LatinHypercubeInitialDesign(scenario, n_configs = self.budget, seed=seed + (i * 100))
            configs += lhd.select_configurations()
            logger.info("Last configurations, added configuration: {}".format(configs[-1]))
        configs = configs[:self.budget]
        logger.info(f"Finalized selection, created {len(configs)} random configurations.")

        # Evaluate configs
        for config in configs:
            self.evaluate_configuration(config)
        logger.info("Random search finished")

     
    def get_objectives(self) -> list[str]:
        """
        Returns the objectives of the experiment.
        """
        return ["error_percentage", "inference_efficiency", "training_efficiency"]
    
     
    def get_crash_costs(self) -> list[float]:
        """
        Returns the crash costs for the objectives.
        """
        return [1.0, math.inf, math.inf]
    
