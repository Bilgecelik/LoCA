import numpy as np
import math
import torch
from torchvision.transforms import transforms
import kornia.augmentation as K
import warnings
from torch.utils.data import random_split
import random
from itertools import cycle

# from avalanche.benchmarks.utils.avalanche_dataset import (
#     AvalancheSubset, AvalancheConcatDataset
# )
from avalanche.benchmarks.utils.classification_dataset import (
    classification_subset, concat_classification_datasets
)
from avalanche.benchmarks import dataset_benchmark
from avalanche.benchmarks.datasets import (
    CIFAR100, TinyImagenet
)

from streams.cir_utils import (
    get_pmf,
    get_per_entity_prob,
    set_random_seed,
    get_dataset_subset_by_class_list
)


##############################################################################
#                              Generator
##############################################################################


def cir_generator(
        train_set,
        #test_set,
        n_e,
        s_e,
        sampler_type="random",
        use_all_samples=False,
        dist_first_occurrence=None,
        dist_recurrence=None,
        p_a=0.0,
        aug_transforms=None,
        train_transform=None,
        seed=0,
):
    """
    :param train_set: train set of the original dataset.
    :param test_set: test set of the original dataset.
    :param n_e: length of the stream (number of experiences).
    :param s_e: number of samples in each experience.
    :param sampler_type: type of the sampler: random or cyclic_random
    :param use_all_samples: whether to enforce using all samples.
    :param dist_first_occurrence: probability distribution of the first-time
        occurrence for each class along the stream.
    :param dist_recurrence: per-class probability of repetition after its first
        occurrence.
    :param p_a: probability of augmentation.
    :param aug_transforms: transformations applied for augmentation.
    :param seed: random seed.
    :return: CL scenario.
    """

    # List classes in the dataset and shuffle them
    classes = list(set(train_set.targets))
    random.Random(seed).shuffle(classes)
    n_classes = len(classes)

    # Set manual random after setting the order of classes
    set_random_seed(seed)

    # Initialize scenario table with zeros
    scenario_table = torch.zeros(n_classes, n_e)

    # ####################################
    #            Dataset info
    # ####################################
    train_targets = torch.LongTensor(train_set.targets)

    def get_class_indices(c):
        indices_c = torch.where(train_targets == c)[0]
        return indices_c


    indices_per_class = {c: get_class_indices(c) for c in range(n_classes)}
    n_samples_per_class = {c: len(get_class_indices(c)) for
                           c in range(n_classes)}

    # ####################################
    #     First occurrence of each class
    # ####################################
    # Calculate PMF for first occurrence
    pmf_probs = get_pmf(n_e, **dist_first_occurrence)
    first_occurrences = np.random.choice(list(range(n_e)), n_classes,
                                         p=pmf_probs, replace=True)

    def assign_first_occurrence(c):
        # Sets first occurrence of a class c in the scenario table
        scenario_table[c, first_occurrences[c]] = 1

    _ = [assign_first_occurrence(c) for c in range(n_classes)]

    # ####################################
    #       Recurrence of each class
    # ####################################
    # Compute probability of repetition for each class
    P_r = get_per_entity_prob(n_classes, **dist_recurrence)

    # Randomly assign recurrence probability to each class
    np.random.shuffle(P_r)

    # For each class find the recurrences according to their
    # recurrence probabilities
    def set_recurrences(c):
        # Sets recurrences for a given class
        remaining = n_e - (first_occurrences[c] + 1)
        rands = torch.rand(remaining)
        mask = rands < P_r[c]
        rands[mask] = 1
        rands[~mask] = 0
        scenario_table[c, first_occurrences[c] + 1:] = rands

    _ = [set_recurrences(c) for c in range(n_classes)]

    # ####################################
    #        Verify scenario table
    # ####################################
    def check_empty_experiences(scenario_table):
        # Find empty experiences and remove them
        empty_exps = torch.where(torch.sum(scenario_table, dim=0) == 0)[0]
        if len(empty_exps) == 0:
            return scenario_table
        else:
            non_empty = torch.where(torch.sum(scenario_table, dim=0) != 0)[0]
            msg = f"\n\nRemoving {len(empty_exps)} empty experiences ...\n" + \
                  f"Current number of experiences: {len(non_empty)}"
            warnings.warn(msg)
            return scenario_table[:, non_empty]

    scenario_table = check_empty_experiences(scenario_table)
    n_e = scenario_table.shape[1]

    # ####################################
    #    Calculate number of samples per class per experience
    # ####################################

    # Compute weights for each class
    # class_weights = {c: torch.sum(scenario_table[c]).item() for c in classes}

    n_samples_table = torch.zeros(n_classes, n_e).long()

    def set_n_samples(exp_id):
        present_classes = torch.where(
            scenario_table[:, exp_id] == 1)[0].numpy()
        np.random.shuffle(present_classes)

        weights = [1 / len(present_classes) for _ in
                   range(len(present_classes))]
        n_samples_per_class = [math.floor(w * s_e) for w in weights]

        # Add the remaining
        remaining = s_e - sum(n_samples_per_class)
        if remaining > 0:
            for i in cycle(list(range(len(n_samples_per_class)))):
                n_samples_per_class[i] += 1
                remaining -= 1
                if remaining == 0:
                    break

        def set_sample_number(c, n):
            n_samples_table[c][exp_id] = n

        _ = [set_sample_number(c, n) for c, n in
             zip(present_classes, n_samples_per_class)]

    # For each experience in the stream, create a dataset
    _ = [set_n_samples(exp_id) for exp_id in range(n_e)]

    # ####################################
    #     Sample indices for each class
    # ####################################

    selected_indices = {}

    def sample_per_class(c):
        selected_indices[c] = {}
        if sampler_type == "random":
            indices_c = indices_per_class[c][torch.randperm(
                len(indices_per_class[c]))]
            exp_c = torch.where(n_samples_table[c] != 0)[0]
            for e, n in zip(exp_c, n_samples_table[c][exp_c]):
                # Shuffle everytime
                indices_c = indices_c[torch.randperm(len(indices_c))]
                selected = indices_c[:n]
                selected_indices[c][e.item()] = selected

        elif sampler_type == "cyclic_random":
            indices_c = indices_per_class[c][torch.randperm(
                len(indices_per_class[c]))]
            exp_c = torch.where(n_samples_table[c] != 0)[0]

            if sum(n_samples_table[c]).item() < n_samples_per_class[c] \
                    and use_all_samples:
                rem = n_samples_per_class[c] - sum(n_samples_table[c]).item()
                n_rem = math.floor(rem / len(exp_c))
                n_rem_per_class = [n_rem] * len(exp_c)
                r_ = rem - sum(n_rem_per_class)
                n_rem_per_class[-1] += r_
                n_samples_table[c][exp_c] += torch.LongTensor(n_rem_per_class)

            shuffle_from = len(indices_c)
            for e, n in zip(exp_c, n_samples_table[c][exp_c]):
                selected = indices_c[:n]
                selected_indices[c][e.item()] = selected
                indices_c = torch.cat([indices_c[n:], selected])
                shuffle_from = max(0, shuffle_from-len(selected))
                indices_c[shuffle_from:] = indices_c[shuffle_from:][torch.randperm(
                    len(indices_c[shuffle_from:]))]
        else:
            raise NotImplementedError()

    _ = [sample_per_class(c) for c in range(n_classes)]

    # ####################################
    #     Create dataset per experience
    # ####################################

    def create_dataset_exp_i(exp_i):
        present_classes = torch.where(scenario_table[:, exp_i] != 0)[0]
        all_indices_i = torch.cat([selected_indices[c.item()][exp_i]
                                   for c in present_classes])
        # Probability of augmentation
        rand_prob = torch.rand(len(all_indices_i))
        indices_augment = torch.where(rand_prob < p_a)[0]
        indices_original = torch.where(rand_prob >= p_a)[0]

        samples_c_aug = all_indices_i[indices_augment]
        samples_c_original = all_indices_i[indices_original]

        final_ds_list = []
        if len(samples_c_aug) != 0:
            ds_i_augment = classification_subset(train_set,
                                                 indices=samples_c_aug.tolist(),
                                                 transform=aug_transforms)
            final_ds_list.append(ds_i_augment)
        if len(samples_c_original) != 0:
            ds_i_original = classification_subset(train_set,
                                                  indices=samples_c_original.tolist(),
                                                  transform=train_transform)
            final_ds_list.append(ds_i_original)

        # Train set
        ds_train_i = concat_classification_datasets(final_ds_list)

        indices = list(range(len(ds_train_i)))
        random.shuffle(indices)
        train_data_len = int(len(ds_train_i)*0.8)

        ds_train_i_final = classification_subset(ds_train_i, indices[:train_data_len])
        ds_train_i_val = classification_subset(ds_train_i, indices[train_data_len:])

        return ds_train_i_final, ds_train_i_val, all_indices_i

    stream_items = [create_dataset_exp_i(i) for i in range(n_e)]
    train_datasets = [t[0] for t in stream_items]
    val_datasets = [t[1] for t in stream_items]
    samples_per_exp = [t[2] for t in stream_items]

    # ####################################
    #           Create benchmark
    # ####################################

    benchmark = dataset_benchmark(
        train_datasets=train_datasets,
        test_datasets=val_datasets,
    )

    # ####################################
    #        Benchmark statistics
    # ####################################

    # -----> Scenario table, n-Samples table and n-TrainSet
    benchmark.first_occurrences = first_occurrences
    benchmark.scenario_table = scenario_table
    benchmark.n_samples_table = n_samples_table
    benchmark.n_trainset = len(train_set)

    # -----> Samples and number of samples in each experience
    benchmark.samples_per_exp = samples_per_exp
    benchmark.n_samples_per_exp = [len(benchmark.samples_per_exp[i]) for i in
                                   range(scenario_table.shape[1])]

    # -----> Total number of samples and unique number of samples in the stream
    all_selected_indices = torch.cat(samples_per_exp)
    all_selected_indices_unique = torch.unique(all_selected_indices)

    not_selected = torch.BoolTensor([True] * len(train_set))
    not_selected[all_selected_indices_unique] = False

    benchmark.n_total_samples = len(all_selected_indices)
    benchmark.n_unique_samples = len(all_selected_indices_unique)

    # Classes in each experience:
    benchmark.present_classes_in_each_exp = [
        torch.where(benchmark.scenario_table[:, i])[0]
        for i in range(n_e)
    ]

    # Seen classes up to each experience
    def classes_seen_sofar(i):
        seen_classes = benchmark.present_classes_in_each_exp[:i + 1]
        seen_classes = set(torch.cat(seen_classes).numpy())
        seen_classes = torch.LongTensor(list(seen_classes))

        return seen_classes

    benchmark.seen_classes = [classes_seen_sofar(i) for i in
                              range(len(benchmark.present_classes_in_each_exp))]
    return benchmark


##############################################################################
#                                 CIFAR-100
##############################################################################


def cir_sampling_cifar100(
        dataset_root,
        n_e: int = 10,
        s_e: int = 500,
        p_a: float = 0.0,
        sampler_type="random",
        use_all_samples=True,
        dist_first_occurrence: dict = None,
        dist_recurrence: dict = None,
        seed: int = 0,
        classes_to_use: list = None
):

    # ----------> Transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
            transforms.Resize((224, 224)),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
            ),
            transforms.Resize((224, 224)),
        ]
    )
    def squeeze_tensor(t):
        return t.squeeze()

    aug_transforms = transforms.Compose([
        transforms.ToTensor(),

        K.ColorJiggle(0.5, 0.1, 0.1, 0.1, p=1.0),
        K.RandomAffine(120, p=1.0),

        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                             std=(0.2673, 0.2564, 0.2762)),
        transforms.Resize((224, 224)),
        squeeze_tensor
    ])

    # ----------> Train and test sets
    train_set = CIFAR100(root=dataset_root, train=True,
                         transform=None, download=True)
    test_set = CIFAR100(root=dataset_root, train=False,
                        transform=eval_transform, download=True)

    # Create subsets if classes_to_use is not None
    if classes_to_use:
        train_set = get_dataset_subset_by_class_list(train_set, classes_to_use)
        test_set = get_dataset_subset_by_class_list(test_set, classes_to_use)


    # ----------> Benchmark
    benchmark = cir_generator(
        train_set,
        #test_set,
        n_e=n_e,
        s_e=s_e,
        p_a=p_a,
        sampler_type=sampler_type,
        use_all_samples=use_all_samples,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        aug_transforms=aug_transforms,
        train_transform=train_transform,
        seed=seed
    )

    return benchmark, test_set


##############################################################################
#                            Tiny-ImageNet
##############################################################################


def cir_sampling_tinyimagenet(
        dataset_root,
        n_e: int = 10,
        s_e: int = 500,
        p_a: float = 0.0,
        sampler_type="random",
        use_all_samples=True,
        dist_first_occurrence: dict = None,
        dist_recurrence: dict = None,
        seed: int = 0,
        classes_to_use: list = None
):
    # ----------> Transformations
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )


    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    def squeeze_tensor(t):
        return t.squeeze()

    aug_transforms = transforms.Compose([
        transforms.ToTensor(),

        K.ColorJiggle(0.5, 0.1, 0.1, 0.1, p=1.0),
        K.RandomAffine(120, p=1.0),

        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409),
                             std=(0.2673, 0.2564, 0.2762)),
        squeeze_tensor
    ])

    # ----------> Train and test sets
    train_set = TinyImagenet(root=dataset_root, train=True,
                             transform=None)
    test_set = TinyImagenet(root=dataset_root, train=False,
                            transform=eval_transform)

    # Create subsets if classes_to_use is not None
    if classes_to_use:
        train_set = get_dataset_subset_by_class_list(train_set, classes_to_use)
        test_set = get_dataset_subset_by_class_list(test_set, classes_to_use)

    # ----------> Benchmark
    benchmark = cir_generator(
        train_set,
        test_set,
        n_e=n_e,
        s_e=s_e,
        p_a=p_a,
        sampler_type=sampler_type,
        use_all_samples=use_all_samples,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        aug_transforms=aug_transforms,
        train_transform=train_transform,
        seed=seed
    )

    return benchmark


##############################################################################
#                                   Test
##############################################################################


if __name__ == "__main__":
    dataset_root = "../../data/"

    dist_first_occurrence = {'dist_type': 'geometric', 'p': 0.1}
    dist_recurrence = {'dist_type': 'fixed', 'p': 0.02}
    # classes_to_use = [23, 8, 11, 7, 48, 13, 1, 91, 94, 54, 16, 63, 52, 41, 80,
    #                   2, 47, 87, 78, 66, 19, 6, 24, 10, 59, 30, 22, 29, 83, 37,
    #                   93, 81, 43, 99, 86, 28, 34, 88, 44, 14, 84, 70, 4, 20, 15,
    #                   21, 31, 76, 57, 67, 73, 50, 69, 25, 98, 46, 96, 0, 72, 35,
    #                   58, 92, 3, 95, 56, 90, 26, 40, 55, 89, 75, 71, 60, 42, 9,
    #                   82, 39, 18, 77, 68, 32, 79, 12, 85, 36, 17, 64, 27, 74, 45]

    benchmark = cir_sampling_cifar100(
        dataset_root=dataset_root,
        n_e=200,
        s_e=500,
        p_a=1.0,
        sampler_type="random",
        use_all_samples=False,
        dist_first_occurrence=dist_first_occurrence,
        dist_recurrence=dist_recurrence,
        seed=0,
        classes_to_use=None,
    )

    for i,d in enumerate(benchmark.train_stream):
        print(d.classes_in_this_experience)
        print(benchmark.test_stream[i].classes_in_this_experience)
        print()

    print(len(benchmark.train_stream))
    print(len(benchmark.test_stream))

    # for i, train_stream in enumerate(benchmark.train_stream):
    #     print(len(benchmark.train_stream[i].classes_in_this_experience))