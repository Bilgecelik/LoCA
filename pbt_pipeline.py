from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import MaximumIterationStopper
from ray import tune, air
import numpy as np
from cl_trainable import CLTrainable
from ray.tune.utils import validate_save_restore
import ray
import argparse

# both of these should return (to catch save/load checkpoint errors before execution)
ray.init()
# validate_save_restore(CLTrainable)
# validate_save_restore(CLTrainable, use_object_store=True)

def pbt_run(
        device: str,
        number_of_experiences: int,
        number_of_steps: int,
        perturb_interval: int,
        number_of_trials: int,
        quantile_fraction: float,
        resample_probability: float,
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

    #tune parameters
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=perturb_interval,
        quantile_fraction=quantile_fraction,
        metric = "mean_loss",
        mode="min",
        resample_probability=resample_probability,
        hyperparam_mutations={
            "lr": tune.qloguniform(5e-3, 1e-1, 5e-4),
            "momentum": tune.qloguniform(5e-3, 1e-1, 5e-4),
        }, #specifies the hps that should be perturbed by PBT and defines the resample distribution for each hp
    )

    if device == 'cpu':
        trainable = CLTrainable
    else:
        trainable = tune.with_resources(CLTrainable, {"gpu": 1})

    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            reuse_actors=True,
            num_samples=number_of_trials,
            max_concurrent_trials=number_of_trials,
        ),
        run_config=air.RunConfig(
            name="pbt_test",
            verbose=3,
            stop=MaximumIterationStopper(max_iter=number_of_steps),
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=2,
                checkpoint_at_end=True,
            ),
        ),
        param_space={
            "lr": tune.uniform(0.001, 1),
            "momentum": tune.uniform(0.001, 1),
            "device": device,
            "number_of_experiences": number_of_experiences,
        }
        )

    results = tuner.fit()
    best_result = results.get_best_result("mean_loss", "min")
    best_loss = best_result.metrics["mean_loss"]
    best_trial_config = best_result.config
    best_checkpoint = best_result.checkpoint

    #Initial tune call
    print(best_loss)
    print(best_trial_config)
    print(CLTrainable.get_auto_filled_metrics())


parser = argparse.ArgumentParser(description='PBT run arguments')

parser.add_argument("-d", "--device",
                    type=str,
                    help="device to use for CL and PBT experiments: 'cpu' or 'gpu'",
                    default='cpu')
parser.add_argument("-e", "--number_of_experiences",
                    type=int,
                    help="number of experiences to generate in CL run.",
                    default=10)
parser.add_argument("-s", "--number_of_steps",
                    type=int,
                    help="number of iterations in PBT experiment.",
                    default=10)
parser.add_argument("-p", "--perturb_interval",
                    type=int,
                    help="#decide whether a trial should continue or exploit a different trial every this many training iterations",
                    default=2)
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


args = parser.parse_args()

pbt_run(
    device = args.device,
    number_of_experiences= args.number_of_experiences,
    number_of_steps=args.number_of_steps,
    perturb_interval=args.perturb_interval,
    number_of_trials=args.number_of_trials,
    quantile_fraction=args.quantile_fraction,
    resample_probability=args.resample_probability,
)
