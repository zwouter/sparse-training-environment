import datetime
import logging 
import hydra
import os
import socket
import json
from omegaconf import DictConfig
import mo_smac
from experiment import Experiment

# Prevent TF warning about oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
# Hide any GPUs form TensorFlow, which are used in some of our dependencies.
# Otherwise, it might reserve memory and make it unavailable to JAX.
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")


# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """
    Starts an experiment using the Hydra configuration in config/config.yaml. All fields can be overwritten by command line arguments.
    Possible experiment types:
    - reproduction              (Perform a reproduction experiment on the default hyperparameter configuration)
    - optimization              (Use MO-SMAC to approximate the Pareto Front in the hyperparameter configuration space)
    - optimization-evaluation   (Re-evaluate the first 2 non-dominated layers from a previous optimization run on the full dataset)
    - evaluation                (Evaluate the 20 configurations with the best average error rates from a previous optimization run using 17 sparsity configurations on the full dataset)
    - random-search             (Perform a random search on the hyperparameter configuration space using Latin Hypercube Sampling)
    """
    logger.info(f"Starting {cfg.experiment.experiment_type} experiment on {cfg.experiment.dataset}")
    logger.info(json.dumps(dict(cfg), indent=4, default=str))

    output_dir = f"{cfg.output_dir}/smac_output/{cfg.experiment.dataset}_{cfg.experiment.experiment_type}"
    scenario_name = f"{datetime.datetime.now().strftime('%m-%d')}" # Add date to distinguish between runs
    output_path = f"{output_dir}/{scenario_name}/{cfg.seed}"

    logger.info(f"Starting {cfg.experiment.experiment_type} experiment")
    with Experiment(cfg.experiment, cfg.seed, output_path) as experiment:
        if cfg.experiment.experiment_type == "reproduction":
            experiment.reproduction_experiment()
        elif cfg.experiment.experiment_type == "optimization":
            experiment.optimization_experiment(output_dir, scenario_name, cfg.seed, cfg.n_workers, cfg.use_seeds, cfg.use_instances)
        elif cfg.experiment.experiment_type == "optimization-evaluation":
            experiment.optimization_experiment_evaluation(cfg.experiment.optimization_path, seed=cfg.seed)
        elif cfg.experiment.experiment_type == "evaluation":
            experiment.evaluation_experiment(cfg.experiment.optimization_path, seed=cfg.seed)
        elif cfg.experiment.experiment_type == "random-search":
            experiment.latin_random_search_experiment(experiment, cfg.seed)
    logger.info(f"Finished {cfg.experiment.experiment_type} experiment")

    
            
def log_system_info():
    """
    Debug logging of system information can be added here.
    """
    logger.info(f"Hostname: {socket.gethostname()}")
    

if __name__ == "__main__":
    log_system_info()
    run_experiment()
