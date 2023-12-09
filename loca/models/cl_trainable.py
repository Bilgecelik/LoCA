# Wrap CL as a trainable for PBT
from ray import tune
import torch
import os

from models.continual_strategy import l2p_strategy

class CLTrainable(tune.Trainable):
    """Train a CL model with Trainable and PopulationBasedTraining
       scheduler.
    """
    def setup(self, config):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and config.get("device") == 'gpu' else 'cpu')
        print(f"Device is {self.device}.")

        self.train_stream = config["train_data"]
        self.validation_stream = config["validation_data"]

        self.strategy = l2p_strategy(device=config.get("device"),
                              num_classes=self.train_data.first_occurrences.shape[0],
                              learning_rate=config.get("lr", 0.03),
                              train_mb_size=config.get("train_mb_size"),
                              train_epochs=config.get("train_epochs"),
                              eval_mb_size=config.get("eval_mb_size"),
                              prompt_pool_size=config.get("prompt_pool_size", 20),
                              prompt_length=config.get("prompt_length", 5),
                              top_k=config.get("top_k", 5),
                              )

        print(self.strategy)
        self.checkpoint_frequency = config.get("checkpoint_frequency")

    def step(self):
        #Search step - pbt step per data point in experience, evaluate on experiences 0-current
        print(f"Training iteration: {self.training_iteration}")
        #print(f"Current strategy at iteration {self.iteration}: {self.strategy}")
        self.strategy.train(self.train_stream[self.training_iteration])

        print(f"Evaluating on experiences: 0 - {self.validation_stream[self.training_iteration]}")
        step_loss = self.strategy.eval(self.validation_stream[:self.training_iteration+1])['Loss_Stream/eval_phase/test_stream/Task000']
        step_accuracy = self.strategy.eval(self.validation_stream[:self.training_iteration+1])['Top1_Acc_Stream/eval_phase/test_stream/Task000']
        step_forgetting = self.strategy.eval(self.validation_stream[:self.training_iteration+1])['StreamForgetting/eval_phase/test_stream']
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

