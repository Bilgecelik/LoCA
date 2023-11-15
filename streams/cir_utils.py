import numpy as np
import math
import random
from scipy.stats import poisson, geom
import torch

from avalanche.benchmarks.utils import AvalancheSubset


def get_pmf(n_entities, dist_type, **kwargs):
    """ Returns the probability mass function (PMF) over set of discrete values.
    """
    if dist_type == "uniform":
        pmf = np.array([1.0] * n_entities)
        value = 1.0 / float(n_entities)
        pmf[:] = value
        return pmf

    elif dist_type == "uniform_alpha":
        pmf = np.array([0.0] * n_entities)
        idx_upper = math.ceil(kwargs["alpha"] * n_entities)
        value = 1.0 / float(idx_upper)
        pmf[:idx_upper] = value
        pmf[:idx_upper] = value
        return pmf

    elif dist_type == "poisson":
        pmf = poisson(mu=kwargs["mu"]).pmf(list(range(n_entities)))
        pmf = pmf / np.sum(pmf)
        return pmf

    elif dist_type == "zipf":
        def zipf_pmf(num_entities, exponent=1.0):
            vals = 1. / (np.arange(1, num_entities + 1)) ** exponent
            return vals / np.sum(vals)

        pmf = zipf_pmf(n_entities, exponent=kwargs["exponent"])
        return pmf

    elif dist_type == "zipf_reversed":
        def zipf_pmf(num_entities, exponent=1.0):
            vals = 1. / (np.arange(1, num_entities + 1)) ** exponent
            return vals / np.sum(vals)

        pmf = zipf_pmf(n_entities, exponent=kwargs["exponent"])
        pmf = list(reversed(pmf))
        return pmf

    elif dist_type == "geometric":
        pmf = geom.pmf(list(range(1, n_entities + 1)), kwargs["p"])
        pmf = pmf / np.sum(pmf)
        return pmf

    else:
        raise NotImplementedError()


def get_per_entity_prob(n_entities, dist_type, **kwargs):
    """ Returns the per-entity probability  over a set of discrete values.
    """
    if dist_type == "random":
        pep = np.random.rand(n_entities)
        return pep

    elif dist_type == "fixed":
        pep = np.array([kwargs["p"]] * n_entities)
        return pep

    elif dist_type == "poisson":
        pep = poisson(mu=kwargs["mu"]).pmf(list(range(n_entities)))
        return pep

    elif dist_type == "zipf":
        def zipf(num_entities, exponent=1.0):
            vals = 1. / (np.arange(1, num_entities + 1)) ** exponent
            return vals

        pep = zipf(n_entities, exponent=kwargs["exponent"])
        return pep

    elif dist_type == "fixed-dual":
        n_low = int(n_entities * kwargs["frac"])
        n_high = n_entities - n_low

        pep_low = [kwargs["p_l"]] * n_low
        pep_high = [kwargs["p_h"]] * n_high

        pep = pep_low + pep_high
        random.shuffle(pep)

        return pep

    else:
        raise NotImplementedError()


def set_random_seed(seed):
    print("Setting random seed: ", seed)
    random.seed(seed)
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False


def get_dataset_subset_by_class_list(
        dataset,
        subset_classes,
        num_samples=None,
        map_classes=True,
) -> AvalancheSubset:
    """ Returns subset of a dataset from a list of classes
    """
    targets = torch.LongTensor(dataset.targets)

    def get_class_idx(c):
        idx_c = torch.where(targets == c)[0]
        if num_samples:
            idx_c = idx_c[torch.randperm(len(idx_c))]
            idx_c = idx_c[:num_samples]

        return idx_c

    all_idx = [get_class_idx(c) for c in subset_classes]
    all_idx = torch.cat(all_idx)

    # Map classes ?
    target_classes = list(range(len(subset_classes)))
    if map_classes:
        class_mapping = {k: v for (k, v) in zip(subset_classes, target_classes)}
    else:
        class_mapping = None

    # Create subset
    subset_ds = AvalancheSubset(dataset, all_idx, class_mapping=class_mapping)

    return subset_ds
