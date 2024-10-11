
import tqdm as tqdm
import copy
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import json
import os
from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as cs_json
# from typing import Self
from typing_extensions import Self


class Run:
    """
    Representation of a single sparse optimization run.
    Each run represents a single configuration of a model, and its objectives after being trained on one seed on a single fold of the data.
    """
    config: dict
    seed: int
    smac_seed: str
    fold: int
    index: int
    error_percentage: float
    inference_flops: float
    training_flops: float
    
    
    def __init__(self, config, seed, smac_seed, fold, index, error_percentage, inference_flops, training_flops) -> None:
        self.config = config
        self.seed = seed
        self.smac_seed = smac_seed
        self.fold = fold
        self.index = index
        self.error_percentage = error_percentage
        self.inference_flops = inference_flops
        self.training_flops = training_flops


    @classmethod
    def from_json_smac(cls, configs: list[dict], run: dict, smac_seed: str, index: int) -> Self:
        """
        Creates a Run object from the fields found in a SMAC runhistory file.
        """
        return cls(
            config=configs[str(run[0])],
            seed=int(run[1]),
            smac_seed=smac_seed,
            fold=run[2],
            index=index,
            error_percentage=run[4][0],
            inference_flops=run[4][1],
            training_flops=run[4][2]
            )


    @classmethod
    def from_json_complete(cls, run: dict, smac_seed: str, index: int) -> Self:
        """
        Creates a Run object from the fields found in an objectives_complete runhistory file.
        """
        return cls(
            config=run['config'],
            seed=run.get('seed', 0),
            smac_seed=smac_seed,
            fold=run.get('instance', 0),
            index=index,
            error_percentage=run['error_percentage'],
            inference_flops=run['inference_flops'],
            training_flops=run['training_flops'],
            )
        
    
    def __str__(self):
        return f"Run object with config: {self.config},\n seed: {self.seed},\n fold: {self.fold},\n smac seed: {self.smac_seed},\n index: {self.index},\n error: {self.error_percentage},\n inference: {self.inference_flops},\n training: {self.training_flops}"
    
    
    def get_config(self) -> dict:
        return self.config
    
    
    def get_objectives(self) -> tuple[float, float, float]:
        return self.error_percentage, self.inference_flops, self.training_flops
    


class RunHistory:
    """
    Representation of one experiment run in which multiple configurations (Runs) are evaluated.
    """
    
    name: str
    configspace: ConfigurationSpace
    hyperparameters: list[dict]
    runs: list[Run]


    def __init__(self, name: str, configspace: ConfigurationSpace, hyperparameters: list[dict], runs: list[Run]) -> None:
        self.name = name
        self.configspace = configspace
        self.hyperparameters = hyperparameters
        self.runs = runs        
    
    
    @classmethod
    def from_json_smac(cls, path: str, name: str = None):
        """
        Creates a Runhistory object from the SMAC output folder.
        """
        # Extract the full path to the folders of all different seeds
        seeds = [f.path for f in os.scandir(path) if f.is_dir()]
        with open(f"{seeds[0]}/configspace.json", "r") as file:
            # Extract the configspace from one of the seeds. It's the same for all of them.
            configspace = file.read()
            hyperparameters = json.loads(configspace)['hyperparameters']
            configspace = cs_json.read(configspace)
        runs = []
        for seed in seeds:
            # Combine the runs of all different seeds for a complete runhistory
            path = f"{seed}/runhistory.json"
            if not os.path.isfile(path):
                continue      
            with open(path, "r") as file:
                runhistory = json.loads(file.read())
                seed_name = os.path.basename(os.path.normpath(seed))
                for i, run in enumerate(runhistory['data']):
                    if int(run[2]) != -1:
                        runs.append(Run.from_json_smac(runhistory['configs'], run, seed_name, i))
                # runs += [Run.from_json_smac(runhistory['configs'], run, seed_name, i) for i, run in enumerate(runhistory['data'])]
                
        if not name:
            name = os.sep.join(os.path.normpath(path).split(os.sep)[-4:-2])
        return cls(name, configspace, hyperparameters, runs)
        
        
    @classmethod
    def from_json_complete(cls, path: str, name: str = None):
        """
        Creates a Run object from an objectives_complete runhistory file.
        Can also be used for experiments ran without SMAC, does need the SMAC configspace json to be copied.
        """
        configspace = None
        runs = []            
        
        # Extract the full path to the folders of all different seeds
        seeds = [f.path for f in os.scandir(path) if f.is_dir()]
        with open(f"{seeds[0]}/configspace.json", "r") as file:
            # Extract the configspace from one of the seeds. It's the same for all of them.
            configspace = file.read()
            hyperparameters = json.loads(configspace)['hyperparameters']
            configspace = cs_json.read(configspace)
        for seed in seeds:
            # Combine the runs of all different seeds for a complete runhistory
            path = f"{seed}/objectives_complete.json"
            if not os.path.isfile(path):
                continue
            with open(path, "r") as file:
                runhistory = json.loads(file.read())
                seed_name = os.path.basename(os.path.normpath(seed))
                for i, run in enumerate(runhistory['results']):
                    runs.append(Run.from_json_complete(run, seed_name, i))
        
        if not name:
            name = os.sep.join(os.path.normpath(path).split(os.sep)[-4:-2]).replace(os.sep, "-")
        return cls(name, configspace, hyperparameters, runs)
    

    def compare_configs(self, config1: dict, config2: dict) -> bool:
        """
        Config id and seed can differ between multiple evaluations of the same config. 
        """
        config1 = config1.copy()
        config2 = config2.copy()
        if 'config_id' in config1:
            del config1['config_id']
        if 'config_id' in config2:
            del config2['config_id']
        if 'seed' in config1:
            del config1['seed']
        if 'seed' in config2:
            del config2['seed']
        return config1 == config2


    def get_configs(self, until_index: int = -1) -> list[dict]:
        """
        Return all unique configurations.
        Configurations contain a config_id. This is actually more comparable to a run id, and differs between multiple evaluations of one config.
        """
        configs = []
        configs_no_id = []
        for run in self.runs:
            if until_index == -1 or run.index <= until_index:
                config = run.get_config().copy()
                config_no_id = config.copy()
                if 'config_id' in config_no_id:
                    del config_no_id['config_id']
                if config_no_id not in configs_no_id:
                    config['seed'] = run.seed
                    configs.append(config)
                    configs_no_id.append(config_no_id)
        return configs
    
    
    def get_runs(self) -> list[Run]:
        """
        Return all runs.
        """
        return self.runs
       
    
    def get_pareto_set(self, until_index: int = -1) -> list[dict]:
        """
        Returns all non-dominated configurations.
        """
        configs = self.get_configs(until_index)
        costs = np.array([self.average_objectives(config) for config in configs])
        
        is_efficient = np.ones(costs.shape[0], dtype = bool)
        for i, c in enumerate(costs):
            is_efficient[i] = np.all(np.any(costs[:i]>c, axis=1)) and np.all(np.any(costs[i+1:]>c, axis=1))
        pareto_set = [config.copy() for config, efficient in zip(configs, is_efficient) if efficient]
        
        return pareto_set


    def get_pareto_front(self, until_index: int = -1) -> list[tuple[float, float, float]]:
        """
        Returns the average objectives of the pareto set.
        """
        pareto_set = self.get_pareto_set(until_index)
        return [self.average_objectives(config) for config in pareto_set]
        
    
    def average_objectives(self, config: dict) -> tuple[float, float, float]:
        """
        Returns the average objectives of a configuration.
        """
        costs = [run.get_objectives() for run in self.runs if self.compare_configs(run.get_config(), config)]
        return tuple(np.mean(costs, axis=0))
    
    
    def get_dominated_configs(self) -> list[dict]:
        """
        Returns all dominated configurations.
        """
        configs = self.get_configs()
        pareto_set = self.get_pareto_set()
        return [config for config in configs if config not in pareto_set]
    
    
    def get_non_dominated_layers(self, n: int = 1) -> list[dict]:
        """
        Returns 
        """
        res = []
        if n >= 1:
            current_history = copy.deepcopy(self)
            res.append(self.get_pareto_set())
            for _ in range(1, n):
                current_history = current_history.new_runhistory_from_configs(current_history.get_dominated_configs())
                res.append(current_history.get_pareto_set())
        return res
    
    
    def new_runhistory_from_configs(self, configs: list[dict]) -> Self:
        """
        Create a new runhistory object with the same configurationspace from a list of configurations.
        """
        new_runs = []
        for run in self.runs:
            for config in configs:
                if self.compare_configs(run.get_config(), config):
                    new_runs.append(run)
                    break
        return RunHistory(self.name, self.configspace, self.hyperparameters, new_runs)
    
        
    def get_complete_dataframe(self) -> pd.DataFrame:
        """
        Returns a dataframe containing all runs and their objectives.
        """
        empty_config = {hyperparam['name']: None for hyperparam in self.hyperparameters}
        empty_config["config_id"] = None
        empty_config["seed"] = None
        empty_config["fold"] = None
        
        res = []
        for run in self.runs:
            full_config = empty_config.copy()
            for key, val in run.config.copy().items():
                full_config[key] = val
            full_config['seed'] = run.seed
            full_config['fold'] = run.fold
            res.append([*run.get_objectives(), *full_config.values()])
        
        return pd.DataFrame(data=res, columns=["Error", "Inference", "Training", *empty_config.keys()])

        
    def get_averaged_dataframe(self, until_index: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns two dataframes containing the average costs and configurations of all configurations and the incumbents.
        Returns: (dominated df, incumbent df)
        """
        average_costs = []
        average_pareto_costs = []
        incumbents = self.get_pareto_set(until_index)
        empty_config = {hyperparam['name']: None for hyperparam in self.hyperparameters}
        empty_config["config_id"] = None
        empty_config["seed"] = None

        for config in self.get_configs(until_index):
            # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
            average_cost = self.average_objectives(config)

            # Create a full config including inactive hyperparameters
            full_config = empty_config.copy()
            for key, val in config.items():
                full_config[key] = val
            
            if config in incumbents:
                average_pareto_costs.append([*average_cost, *full_config.values()])
            else:
                average_costs.append([*average_cost, *full_config.values()])
        
        dominated_df = pd.DataFrame(data=average_costs, columns=["Error", "Inference", "Training", *empty_config.keys()])
        incumbent_df = pd.DataFrame(data=average_pareto_costs, columns=["Error", "Inference", "Training", *empty_config.keys()])
        dominated_df['Class'] = "Dominated configuration"
        incumbent_df['Class'] = "Incumbent"
        
        return pd.concat([incumbent_df, dominated_df], ignore_index=True)
