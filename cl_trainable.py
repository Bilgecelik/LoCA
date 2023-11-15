# Wrap CL as a trainable for PBT
from ray import tune
import torch
import os

from torch.nn import CrossEntropyLoss
from avalanche.training.supervised.l2p import LearningToPrompt
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from streams.cir_sampling import cir_sampling_cifar100

class CLTrainable(tune.Trainable):
    """Train a CL model with Trainable and PopulationBasedTraining
       scheduler.
    """
    def setup(self, config):
        #self.device = torch.device("cuda:0" if config.get("device") == "gpu" else "cpu")
        self.device = torch.device("cuda:0")
        print(f"Device is {self.device}.")
        print(f"Cuda is available: {torch.cuda.is_available()}")
        self.benchmark, self.test_stream = cir_sampling_cifar100(
            dataset_root="../../Dataset/",
            n_e=config.get("number_of_experiences"),
            s_e=500,
            p_a=1.0,
            sampler_type="random",
            use_all_samples=False,
            dist_first_occurrence={'dist_type': 'geometric', 'p': 0.1},
            dist_recurrence= {'dist_type': 'fixed', 'p': 0.02},  #fixed repetition probability for all classes to have a balanced stream,
            seed=0,
            classes_to_use=None,
        )
        print(self.benchmark.train_stream)

        self.criterion = CrossEntropyLoss()

        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
            forgetting_metrics(experience=True, stream=True),
            strict_checks=False,
        )

        self.strategy = LearningToPrompt(
            model_name='vit_large_patch16_224',
            criterion=CrossEntropyLoss(),
            train_mb_size=config.get("train_mb_size"),
            train_epochs=config.get("train_epochs"),
            eval_mb_size=config.get("eval_mb_size"),
            device=self.device,
            evaluator=self.eval_plugin,
            num_classes=self.benchmark.first_occurrences.shape[0],  #total # of classes in all tasks
            use_vit=True,
            lr=config.get("lr", 0.03),
            pool_size=config.get("prompt_pool_size", 20),
            prompt_length=config.get("prompt_length", 5),
            top_k=config.get("top_k", 5),
            sim_coefficient=0.5, #default in avalanche is 0.1, paper is 0.5, not sensitive
        )
        print(self.strategy)
        self.best_score = 0
        self.checkpoint_frequency = config.get("checkpoint_frequency")

    def step(self):
        #Search step - progress each trial with new experience and new model setup, evaluate on experiences 0-current
        print(f"Training on current experience: {self.benchmark.train_stream[self.iteration].current_experience}")
        print(f"Current strategy at experience {self.benchmark.train_stream[self.iteration].current_experience}: {self.strategy}")
        self.strategy.train(self.benchmark.train_stream[self.iteration])  # this needs to take next experience, not whole benchmark

        print(f"Evaluating on experiences: 0 - {self.benchmark.test_stream[self.iteration].current_experience}")
        step_loss = self.strategy.eval(self.benchmark.test_stream[:self.iteration+1])['Loss_Stream/eval_phase/test_stream/Task000']
        step_accuracy = self.strategy.eval(self.benchmark.test_stream[:self.iteration+1])['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        step_forgetting = self.strategy.eval(self.benchmark.test_stream[:self.iteration+1])['StreamForgetting/eval_phase/test_stream']
        #
        # if self.training_step % self.checkpoint_frequency == 0:
        #     self.save_checkpoint(self.checkpoint_dir)

        return {"mean_loss": step_loss,
                "mean_accuracy": step_accuracy,
                "mean_forgetting": step_forgetting,
                "current_experience": self.iteration}

    def save_checkpoint(self, tmp_checkpoint_dir):
        print(f"Checkpoint saved at training iteration {self.training_iteration} with trial {self.trial_id}.")
        checkpoint_path = os.path.join(tmp_checkpoint_dir, "model.pth")
        torch.save({
            "model": self.strategy.model.state_dict(),
        }, checkpoint_path)
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        print(f"Checkpoint loaded at training iteration {self.training_iteration} with trial {self.trial_id}.")
        checkpoint = torch.load(checkpoint_path)
        self.strategy.model.load_state_dict(checkpoint["model"])

    def _export_model(self, export_formats, export_dir):
        path = os.path.join(export_dir, "exported_model.pt")
        torch.save(self.strategy.model.state_dict(), path)
        return {path}

