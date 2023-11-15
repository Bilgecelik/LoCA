from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import MaximumIterationStopper
from ray import tune, air
from cl_trainable import CLTrainable
from ray.air.integrations.wandb import WandbLoggerCallback
import ray
import argparse
import numpy as np
import random
from ray.tune.utils import validate_save_restore
WANDB_API_KEY='71b542c3072e07c51d1184841ffc50858ab2090e'

random.seed(1234)
np.random.seed(5678)

ray.init()


def pbt_run(
        device: str,
        number_of_experiences: int,
        number_of_steps: int,
        perturb_interval: int,
        number_of_trials: int,
        quantile_fraction: float,
        resample_probability: float,
        search_criterion: str,
        search_mode: str,
        description: str,
        cl_train_mb_size: int,
        cl_train_epochs: int,
        cl_eval_mb_size: int,
        checkpoint_frequency: int,
    ):

    """
    :param resample_probability:
    :param quantile_fraction:
    :param device:
    :param number_of_experiences:
    :param number_of_steps:
    :param perturb_interval:
    :param number_of_trials:
    """

    # # Define trainable class based on device
    trainable = CLTrainable

    # both of these should return (to catch save/load checkpoint errors before execution)
    # validate_save_restore(CLTrainable)
    # validate_save_restore(CLTrainable, use_object_store=True)

    #search space and other configurations
    config = {
            "lr": 0.01,
            "prompt_pool_size": 20,
            "prompt_length": 5,
            "top_k": 5,
            "device": device,
            "number_of_experiences": number_of_experiences,
            "train_mb_size": cl_train_mb_size,
            "train_epochs": cl_train_epochs,
            "eval_mb_size": cl_eval_mb_size,
            "wandb": {
                "project": "OACL Trials",
                "group": trainable.trial_id,
            },
            "checkpoint_frequency": checkpoint_frequency,
            "number_of_trials": number_of_trials,
        }
    #tune parameters
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturb_interval,
        quantile_fraction=quantile_fraction,
        metric = search_criterion,
        mode=search_mode,
        resample_probability=resample_probability,
        hyperparam_mutations={
            "lr": tune.loguniform(0.01, 0.1),
            "prompt_pool_size":tune.randint(5,30),
            "prompt_length":tune.randint(1,20),
            "top_k":tune.randint(1,10),
        }, #specifies the hps that should be perturbed by PBT and defines the resample distribution for each hp
        synch=True,
    )

    # Use ray.tune.run() for asynchronous optimization
    analysis = tune.run(
        CLTrainable,
        name=f"PBT_CL_Experiments_{description}",
        config=config,
        scheduler=scheduler,
        #reuse_actors=True,
        num_samples=number_of_trials,
        max_concurrent_trials=number_of_trials,
        max_failures=1,  # Number of failures before terminating
        local_dir="./ray_results",  # Directory for storing logs and checkpoints
        verbose=3,
        stop=tune.stopper.MaximumIterationStopper(max_iter=number_of_steps),
        checkpoint_freq=min(perturb_interval, checkpoint_frequency),
        checkpoint_at_end=True,
        callbacks=[ray.air.integrations.wandb.WandbLoggerCallback(project="OACL Trials", log_config=True),
                   # callback_on_iteration,
                   # select_best_trial,
        ],
        resources_per_trial={"gpu": 1},
    )

    print(analysis.results)

    # Shutdown Ray
    ray.shutdown()

    # # Access the callback data for each iteration
    # for iteration, data in enumerate(callback_iteration_list, start=1):
    #     print(f"Iteration {iteration} - Callback Data:")
    #     print(data)
    #     print("")
    #
    # # Access the best trial data for each iteration
    # for iteration, data in enumerate(best_trial_list, start=1):
    #     print(f"Iteration {iteration} - Callback Data:")
    #     print(data)
    #     print("")

parser = argparse.ArgumentParser(description='PBT run arguments')

parser.add_argument("-d", "--device",
                    type=str,
                    help="device to use for CL and PBT experiments: 'cpu' or 'gpu'",
                    default='cpu')
parser.add_argument("-e", "--number_of_experiences",
                    type=int,
                    help="number of experiences to generate in CL run.",
                    default=5)
parser.add_argument("-s", "--number_of_steps",
                    type=int,
                    help="number of iterations in PBT experiment.",
                    default=10)
parser.add_argument("-p", "--perturb_interval",
                    type=int,
                    help="decide whether a trial should continue or exploit a different trial every this many training iterations",
                    default=5)
parser.add_argument("-t", "--number_of_trials",
                    type=int,
                    help="number of parallel trials in PBT run.",
                    default=5
                    )
parser.add_argument("-q", "--quantile_fraction",
                    type=float,
                    help="top performing trials percentage",
                    default=0.5
                    )
parser.add_argument("-r", "--resample_probability",
                    type=float,
                    help="resampling and mutation both happen with this probability",
                    default=0.5
                    )
parser.add_argument("-de", "--description",
                    type=str,
                    help="short description of the experiment setup",
                    default='no_description'
                    )
parser.add_argument("-tmb", "--cl_train_mb_size",
                    type=int,
                    help="mini-batch size for cl training",
                    default=32
                    )
parser.add_argument("-emb", "--cl_eval_mb_size",
                    type=int,
                    help="mini-batch size for cl evaluation",
                    default=32
                    )
parser.add_argument("-ep", "--cl_train_epochs",
                    type=int,
                    help="number of cl training epochs at each step",
                    default=1
                    )
parser.add_argument("-c", "--search_criterion",
                    type=str,
                    help="metric to optimize in search algorithm",
                    default='mean_loss'
                    )
parser.add_argument("-m", "--search_mode",
                    type=str,
                    help="search optimization direction: min or max",
                    default='min'
                    )
parser.add_argument("-cf", "--checkpoint_frequency",
                    type=int,
                    help="number of steps between checkpointing, good to set to the perturbation interval",
                    default=5,
                    )

args = parser.parse_args()
print(args)

best_score = 0
pbt_run(
    device = args.device,
    number_of_experiences= args.number_of_experiences,
    number_of_steps=args.number_of_steps,
    perturb_interval=args.perturb_interval,
    number_of_trials=args.number_of_trials,
    quantile_fraction=args.quantile_fraction,
    resample_probability=args.resample_probability,
    description=args.description,
    cl_train_mb_size=args.cl_train_mb_size,
    cl_eval_mb_size=args.cl_eval_mb_size,
    cl_train_epochs=args.cl_train_epochs,
    search_criterion=args.search_criterion,
    search_mode = args.search_mode,
    checkpoint_frequency=args.checkpoint_frequency,
)
