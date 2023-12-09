from ray.tune.schedulers import PopulationBasedTraining
from ray import tune
import ray

import numpy as np
import random
from typing import Dict, Any
from ray.tune.utils import validate_save_restore
WANDB_API_KEY='71b542c3072e07c51d1184841ffc50858ab2090e'

random.seed(1234)
np.random.seed(5678)

ray.init()
def setup_pbt(
        pbt_config: Dict
) -> Any:
    #setup pbt search
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=pbt_config["perturb_interval"],
        quantile_fraction=pbt_config["quantile_fraction"],
        metric = pbt_config["search_criterion"],
        mode=pbt_config["search_mode"],
        resample_probability=pbt_config["resample_probability"],
        hyperparam_mutations={
            "lr": tune.loguniform(0.01, 0.1),
            "prompt_pool_size":tune.randint(5,30),
            "prompt_length":tune.randint(1,20),
            "top_k":tune.randint(1,10),
        }, #specifies the hps that should be perturbed by PBT and defines the resample distribution for each hp
        synch=True,
    )

    return scheduler

def pbt_search(
        train_data,
        validation_data,
        scheduler,
        search_config: Dict,
        pbt_config: Dict,
        trainable,
    ):

    # both of these should return (to catch save/load checkpoint errors before execution)
    validate_save_restore(trainable)
    validate_save_restore(trainable, use_object_store=True)

    #add recent data to trainable config
    search_config["train_data"] = train_data
    search_config["validation_data"] = validation_data

    # Use ray.tune.run() for asynchronous optimization
    analysis = tune.run(
        trainable,
        name=f"PBT_CL_Experiments_{pbt_config['description']}",
        config=search_config,
        scheduler=scheduler,
        #reuse_actors=True,
        num_samples=pbt_config['number_of_trials'],
        max_concurrent_trials=pbt_config['number_of_trials'],
        max_failures=1,  # Number of failures before terminating
        local_dir="./ray_results",  # Directory for storing logs and checkpoints
        verbose=3,
        stop=tune.stopper.MaximumIterationStopper(max_iter=pbt_config['number_of_steps']),
        checkpoint_freq=min(pbt_config["perturb_interval"], pbt_config["checkpoint_frequency"]),
        checkpoint_at_end=True,
        resources_per_trial={"gpu": 1},
    )

    print(analysis.results)

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial(metric="accuracy", mode="max", scope="all")
    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(best_trial, metric="accuracy")

    return best_checkpoint


