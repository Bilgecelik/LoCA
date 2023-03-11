# CL Setup
import torch
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics
from avalanche.training.plugins import EvaluationPlugin
import wandb
from typing import Dict
WANDB_API_KEY='71b542c3072e07c51d1184841ffc50858ab2090e'


def cl_run(
        device: str,
        number_of_experiences: int,
        config: Dict,
) -> None:

    device = torch.device("cuda:0" if device == "gpu" else "cpu")
    benchmark = PermutedMNIST(n_experiences=number_of_experiences, seed=1)

    # set criteria, optimizer (config hp's), model (checkpoint)
    criterion = CrossEntropyLoss()
    model = SimpleMLP(num_classes=10)
    optimizer = SGD(model.parameters(),
                    lr=config["lr"],
                    momentum=config["momentum"])

    # plot with wandb
    wandb_logger = WandBLogger(project_name="OACL Trials",
                               run_name="CL_Solo",
                               log_artifacts=True,
                               config={
                                   "dataset": benchmark,
                                   "model": model,
                                   "strategy": "Naive"
                               }
                               )

    # evaluation plugin
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        loggers=[wandb_logger, InteractiveLogger()],
        strict_checks=False
    )
    # define strategy
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=50,
        train_epochs=5,
        eval_mb_size=10,
        device=device,
        evaluator=eval_plugin
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in benchmark.train_stream:
        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)
        print('Training completed')

        print('Evaluating on experiences until current one.')
        # test also returns a dictionary which contains all the metric values
        print(cl_strategy.eval(benchmark.test_stream[:experience.current_experience + 1]))

    wandb.finish()

parser = argparse.ArgumentParser(description='PBT run arguments')

parser.add_argument("-d", "--device",
                    type=str,
                    help="device to use for CL and PBT experiments: 'cpu' or 'gpu'",
                    default='cpu')
parser.add_argument("-e", "--number_of_experiences",
                    type=int,
                    help="number of experiences to generate in CL run.",
                    default=5)
parser.add_argument("--optimal_lr",
                    type=float,
                    help="optimal learning rate for cl run",
                    default=0.001)
parser.add_argument("--optimal_momentum",
                    type=float,
                    help="optimal momentum for cl run",
                    default=0.9)


args = parser.parse_args()

optimal_config = {"lr": args.optimal_lr,
                  "momentum": args.optimal_momentum}

cl_run(
    device = args.device,
    number_of_experiences= args.number_of_experiences,
    config= optimal_config
)

