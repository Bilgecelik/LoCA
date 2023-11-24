#Setup CL run
#Call CL run in loop
#Check drift
#If drift / interval, call PBT run (setup in PBT_run file) with prompt weights checkpoint and last data
#Get PBT run result (best_trial) and compare with current pipeline
#Continue CL with the best setup from next data point

from data import make_stream
from models.search import pbt_run
from models import continual_strategy
import argparse
import sys
import wandb
WANDB_API_KEY='71b542c3072e07c51d1184841ffc50858ab2090e'
wandb.login()

def main():
    print(sys.argv)
    parser = argparse.ArgumentParser(description='PBT run arguments')

    parser.add_argument("-da", "--data",
                        type=str,
                        help="input data to make continual stream - currently only supports CIR with 'cifar100'",
                        default='cifar100'
                        )
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

    #generate stream
    benchmark, test_stream = make_stream(number_of_experiences = args.number_of_experiences,
                                         s_e = args.s_e,
                                         p_a = args.p_a)

    #setup search config
    config = {
            "lr": 0.01,
            "prompt_pool_size": 20,
            "prompt_length": 5,
            "top_k": 5,
            "device": args.device,
            "number_of_experiences": args.number_of_experiences,
            "train_mb_size": args.cl_train_mb_size,
            "train_epochs": args.cl_train_epochs,
            "eval_mb_size": args.cl_eval_mb_size,
            "checkpoint_frequency": args.checkpoint_frequency,
            "number_of_trials": args.number_of_trials,
        }

    #setup defaults for hps
    learning_rate = config['lr']
    prompt_pool_size = config['prompt_pool_size']
    prompt_length = config['prompt_length']
    top_k = config['top_k']

    #setup strategy
    strategy = continual_strategy(device=args.device,
                                  num_classes=benchmark.first_occurrences.shape[0],
                                  learning_rate=learning_rate,
                                  train_mb_size=args.train_mb_size,
                                  train_epochs=args.train_epochs,
                                  eval_mb_size=args.eval_mb_size,
                                  prompt_pool_size=prompt_pool_size,
                                  prompt_length=prompt_length,
                                  top_k=top_k,
                                  )

    #setup logging
    run = wandb.init(
        # Set the project where this run will be logged
        project="LOCA-Main Flow",
        # Track hyperparameters and run metadata
        config=config)

    #CL flow
    #can strategy start from checkpoint for prompt weights with new hp's?

    results = []
    for experience in benchmark.train_stream:
        #cl flow
        print('Starting experiment...')

        # train returns a dictionary which contains all the metric values
        print(f"Training on current experience: {experience.current_experience}")
        strategy.train(experience)

        print(f"Evaluating on experiences: 0 - {experience.current_experience}")

        loss = strategy.eval(test_stream[:experience.current_experience + 1])['Loss_Stream/eval_phase/test_stream/Task000']
        accuracy = strategy.eval(test_stream[:experience.current_experience + 1])[
            'Top1_Acc_Stream/eval_phase/test_stream/Task000']
        forgetting = strategy.eval(test_stream[:experience.current_experience + 1])[
            'StreamForgetting/eval_phase/test_stream']
        results.append({'experience': experience.current_experience, 'loss': loss, 'accuracy': accuracy, 'forgetting': forgetting})
        wandb.log(results[-1])

    wandb.finish()


    ### to be added to the loop
    pbt_run(
        device=args.device,
        number_of_experiences=args.number_of_experiences,
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
        search_mode=args.search_mode,
        checkpoint_frequency=args.checkpoint_frequency,
    )


if __name__ == "__main__":
    main()


