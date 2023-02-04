from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.stopper import MaximumIterationStopper
from ray import tune
import numpy as np
from cl_trainable import CLTrainable

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    metric = "mean_loss",
    mode="min",
    perturbation_interval=2,
    hyperparam_mutations={
        # distribution for resampling
        "lr": lambda: np.random.uniform(0.0001, 1),
        # allow perturbations within this set of categorical values
        "momentum": [0.8, 0.9, 0.99],
    },
  )

analysis = tune.run(
        CLTrainable,
        name="pbt_test",
        scheduler=scheduler,
        reuse_actors=True,
        verbose=1,
        stop=MaximumIterationStopper(max_iter=10),
        checkpoint_score_attr="mean_loss",
        checkpoint_freq=2,
        keep_checkpoints_num=3,
        num_samples=3,
        config={
            "lr": tune.uniform(0.001, 1),
            "momentum": tune.uniform(0.001, 1),
        },
        resources_per_trial={"cpu": 2, "gpu": 0})

print('best config:', analysis.get_best_config("mean_loss"))