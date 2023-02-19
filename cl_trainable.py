# Wrap CL as a trainable for PBT
from ray import tune
import torch
import os
from typing import Dict, Any, Callable, Optional

from ray.tune.syncer import Syncer
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import PermutedMNIST  # , RotatedMNIST, SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics
from avalanche.training.plugins import EvaluationPlugin

WANDB_API_KEY='71b542c3072e07c51d1184841ffc50858ab2090e'

class CLTrainable(tune.Trainable):
    """Train a CL model with Trainable and PopulationBasedTraining
       scheduler.
    """


    def setup(self, config):
        self.device = torch.device("cuda:0" if config.get("device") == "gpu" else "cpu")
        self.benchmark = PermutedMNIST(n_experiences=config.get("number_of_experiences"), seed=1)
        self.criterion = CrossEntropyLoss()
        self.model = SimpleMLP(num_classes=10)  # start with new model
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.get("lr", 0.01),
            momentum=config.get("momentum", 0.9)
        )
        wandb_logger = WandBLogger(project_name="OACL-Initial Experiments",
                                   run_name="OACL Naive Model - PermutedMNIST - PBT - "+self.trial_id,
                                   log_artifacts=True,
                                   config={
                                       "dataset": self.benchmark,
                                       "model": self.model,
                                       "strategy": "Naive"
                                   }
                                   )
        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            strict_checks=False,
            loggers=[wandb_logger, InteractiveLogger()],
        )
        self.strategy = Naive(
            self.model,
            self.optimizer,
            self.criterion,
            train_mb_size=50,
            train_epochs=5,
            eval_mb_size=10,
            device=self.device,
            evaluator=self.eval_plugin
        )
        self.experience = 0

    def step(self):
        # step in CL is training on one experience, train on first half of task, evaluate on second half
        self.strategy.train(self.benchmark.train_stream[self.experience])  # this needs to take next experience, not whole benchmark
        print(f"Step is {self.experience} and current experience is:")
        print(self.benchmark.train_stream[self.experience].current_experience)
        step_loss = self.strategy.eval(self.benchmark.test_stream[0:self.experience])['Loss_Stream/eval_phase/test_stream/Task000']
        self.experience += 1
        return {"mean_loss": step_loss}

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

    def _export_model(self, export_formats, export_dir):
        path = os.path.join(export_dir, "exported_model.pt")
        torch.save(self.model.state_dict(), path)
        return {path}

    def reset_config(self, new_config):
        for param_group in self.optimizer.param_groups:
            if "lr" in new_config:
                param_group["lr"] = new_config["lr"]
            if "momentum" in new_config:
                param_group["momentum"] = new_config["momentum"]

        self.config = new_config
        self.get_auto_filled_metrics()
        return True

