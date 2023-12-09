# CL Setup
import torch
from torch.nn import CrossEntropyLoss
from avalanche.training.supervised.l2p import LearningToPrompt
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from typing import Any

def l2p_strategy(
        device: str,
        num_classes: int,
        learning_rate: float,
        train_mb_size: int,
        train_epochs: int,
        eval_mb_size: int,
        prompt_pool_size: int,
        prompt_length: int,
        top_k: int,
    ) -> Any:

    device = torch.device("cuda:0" if device == "gpu" else "cpu")

    wandb_logger = WandBLogger(project_name="LOCA Trials",
                               run_name="CL_Solo",
                               log_artifacts=True,
                               config={
                                   "dataset": benchmark,
                                   "model": 'vit_large_patch16_224',
                                   "strategy": "C2P"
                               }
                               )
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[wandb_logger, InteractiveLogger()],
        strict_checks=False,
    )

    strategy = LearningToPrompt(
        model_name='vit_large_patch16_224',
        criterion=CrossEntropyLoss(),
        train_mb_size=train_mb_size,
        train_epochs=train_epochs,
        eval_mb_size=eval_mb_size,
        device=device,
        evaluator=eval_plugin,
        num_classes=num_classes,  # total # of classes in all tasks
        use_vit=True,
        lr=learning_rate,
        pool_size=prompt_pool_size,
        prompt_length=prompt_length,
        top_k=top_k,
        sim_coefficient=0.5,  # default in avalanche is 0.1, paper is 0.5, not sensitive
    )

    return strategy

