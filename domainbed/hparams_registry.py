import numpy as np
from domainbed.lib import misc


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.    
    _hparam('data_augmentation', True, lambda r: True)
    if dataset in ['VLCS','PACS']:
        _hparam('resnet18', True, lambda r: True)
    else:
        _hparam('resnet18', False, lambda r: False)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))
    _hparam('class_balanced', False, lambda r: False)
    _hparam('weight_decay', 0., lambda r: 0.)
    # Nonlinear classifiers disabled
    if dataset in ["RotatedMNIST"]:
        _hparam('nonlinear_classifier', False,
            lambda r: bool(r.choice([True, False])))
    else:
        _hparam('nonlinear_classifier', True,
            lambda r: bool(r.choice([True, False])))

    # Algorithm-specific hparam definitions. Each block of code below
    # corresponds to exactly one algorithm.

    if algorithm == "FedIIR":
        if dataset in ["RotatedMNIST"]:
            _hparam('penalty', 1e-2, lambda r: r.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
        elif dataset in ["VLCS"]:
            _hparam('penalty', 5e-3, lambda r: r.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
        elif dataset in ["PACS"]:
            _hparam('penalty', 1e-3, lambda r: r.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
        elif dataset in ["OfficeHome"]:
            _hparam('penalty', 5e-4, lambda r: r.choice([1e-2, 5e-3, 1e-3, 5e-4, 1e-4]))
        else:
            _hparam('penalty', 1e-3, lambda r: 10**r.uniform(-2, -4))
        _hparam('ema', 0.95, lambda r: r.choice([0.90, 0.95, 0.99]))  


    # Dataset-specific hparam definitions. Each block of code
    # below corresponds to exactly one hparam. Avoid nested conditionals.


    if dataset  == "RotatedMNIST":
        _hparam('lr', 1e-2, lambda r: r.choice([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]))
        _hparam('batch_size', 64, lambda r: r.choice([32, 64]))
    elif dataset  == "VLCS":
        _hparam('lr', 1e-3, lambda r: r.choice([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]))
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))
    elif dataset  == "PACS":
        _hparam('lr', 2.5e-3, lambda r: r.choice([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]))
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))
    elif dataset  == "OfficeHome":
        _hparam('lr', 1e-3, lambda r: r.choice([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]))
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))
    else:
        _hparam('lr', 1e-3, lambda r: r.choice([1e-2, 5e-3, 2.5e-3, 1e-3, 5e-4, 2.5e-4, 1e-4]))
        _hparam('batch_size', 32, lambda r: r.choice([32, 64]))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
