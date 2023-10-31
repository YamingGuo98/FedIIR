import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
import csv

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FiniteDataLoader, FastDataLoader
from domainbed.lib.misc import get_free_gpu

if __name__ == "__main__":
    # parameter settings
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--device_name', type=str, default = None)
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--dataset', type=str, default="RotatedMNIST", choices=['RotatedMNIST', 'VLCS', 'PACS', 'OfficeHome'])
    parser.add_argument('--algorithm', type=str, default="FedIIR")
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--checkpoint_freq', type=int, default=10,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--holdout_fraction', type=float, default=0.1, help='Fraction of each domain set holdout for validation')
    parser.add_argument('--skip_model_save', action='store_true')

    parser.add_argument('--num_clients', type=int, default=10, help='Number of total clients')
    parser.add_argument('--sample_num', type=int, default=3, help='Number of sampled clients in one communication round')
    parser.add_argument('--global_rounds', type=int, default=100, help='Number of global communication rounds')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local update epochs')
    parser.add_argument('--train_index', type=str, default="run_0")
    parser.add_argument('--output_dir', type=str, default="train_output") 
    args = parser.parse_args()

    # the number of sampled clients for RotatedMNIST is set to 5
    if args.dataset in ["RotatedMNIST"]:
        args.sample_num = 5

    # create output files
    dir = os.path.join(args.output_dir, args.dataset, args.algorithm)
    os.makedirs(dir, exist_ok=True)
    stdout_file = os.path.join(args.output_dir, args.dataset, args.algorithm, f"{args.train_index}_stdout")
    stderr_file = os.path.join(args.output_dir, args.dataset, args.algorithm, f"{args.train_index}_stderr")
    with open(stdout_file, "w") as out_file, open(stderr_file, "w") as err_file:
        out_file.write('Begin\n')
        err_file.write('Begin\n')
    sys.stdout = misc.Tee(stdout_file)
    sys.stderr = misc.Tee(stderr_file)

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
        misc.seed_hash(args.hparams_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # set device
    if args.device_name == None:
        if torch.cuda.is_available():
            args.device = torch.device(f"cuda:{get_free_gpu()}")
        else:
            args.device = torch.device("cpu")
    else:
        args.device = torch.device(args.device_name)

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # save results dictionary
    result_dict = {
        "Dataset" : args.dataset,
        "Algorithm": args.algorithm,
        "Num_clients": args.num_clients,
        "Trial_seed": args.trial_seed,
        "Seed": args.seed
    }
    key_list = ["class_balanced", "data_augmentation", "nonlinear_classifier", "resnet18", "resnet_dropout", "weight_decay"]
    for key in hparams:
        if key not in key_list:
            result_dict[key] = hparams[key]
    result_file = "{}_{}_{}.csv".format(args.dataset, args.algorithm, "results")
    with open(os.path.join(args.output_dir, args.dataset, args.algorithm, result_file), mode='a', newline='') as csv_file:
        fieldnames = list(result_dict.keys())
        fieldnames.extend(["Test_env", "Train_acc", "Test_acc"])
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            writer.writeheader()
    
    domains = []
    avg_acc = np.zeros(2)
    
    # execute training
    for test_envs in range(datasets.DATASETS_DICT[args.dataset]):
        args.test_envs = [test_envs]
        print('Args:')
        for k, v in sorted(vars(args).items()):
            print('\t{}: {}'.format(k, v))
        print('HParams:')
        for k, v in sorted(hparams.items()):
            print('\t{}: {}'.format(k, v))

        if args.dataset in vars(datasets):
            dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
        else:
            raise NotImplementedError
        
        # leave one domain as test
        test_data = dataset[args.test_envs[0]]
        test_name = dataset.ENVIRONMENTS[args.test_envs[0]]
        domains.append(test_name)
        test_data, _ = misc.split_dataset(test_data, int(len(test_data)), 
                                                misc.seed_hash(args.trial_seed, args.test_envs[0]))
        train_data = [dataset[i] for i in range(len(dataset)) if i not in args.test_envs]
        # split to client
        num_examples_per_domain = np.zeros(len(train_data))
        for i, env in enumerate(train_data):
            num_examples_per_domain[i] = len(env)*(1 - args.holdout_fraction)
        per_domain_clients = np.ones(len(train_data))
        for i in range(args.num_clients - len(train_data)):
            idx = np.argmax(np.divide(num_examples_per_domain, per_domain_clients))
            per_domain_clients[idx] += 1
        per_domain_clients = per_domain_clients.astype(int)
        per_client_examples = np.floor(np.divide(num_examples_per_domain,per_domain_clients),)

        in_splits = []
        out_splits = []
        test_splits = [(test_data,None)]
        for env_i, env in enumerate(train_data):

            out, in_ = misc.split_dataset(env,
                int(len(env)*args.holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            
            in_for_client = misc.split_dataset_fl(in_, per_domain_clients[env_i], 
                                                per_client_examples[env_i], misc.seed_hash(args.trial_seed, env_i))
            for _, in_client in enumerate(in_for_client):
                in_splits.append((in_client, None))
            out_splits.append((out, None))

        train_loaders = [FiniteDataLoader(
            dataset=env,
            weights=env_weights,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
            for _, (env, env_weights) in enumerate(in_splits)]
        eval_loaders = [FastDataLoader(
            dataset=env,
            batch_size=64,
            num_workers=dataset.N_WORKERS)
            for env, _ in (out_splits + test_splits)]
        eval_weights = [None for _, weights in (out_splits + test_splits)]
        eval_loader_names = ['env{}_out'.format(i)
                            for i in range(len(out_splits))]
        eval_loader_names += ['test']

        algorithm_dict = None
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
            len(dataset) - len(args.test_envs), hparams, args)

        if algorithm_dict is not None:
            algorithm.load_state_dict(algorithm_dict)

        algorithm.to(args.device)

        checkpoint_vals = collections.defaultdict(lambda: [])

        checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

        def save_checkpoint(filename):
            if args.skip_model_save:
                return
            save_dict = {
                "args": vars(args),
                "model_input_shape": dataset.input_shape,
                "model_num_classes": dataset.num_classes,
                "model_num_domains": len(dataset) - len(args.test_envs),
                "model_hparams": hparams,
                "model_dict": algorithm.state_dict()
            }
            torch.save(save_dict, os.path.join(args.output_dir, args.dataset, args.algorithm, filename))

        epochs_count = []
        last_results_keys = None
        best_train_acc = 0
        best_test_acc = 0
        for epoch in range(args.global_rounds):
            # sampling clients
            sampled_loaders = []
            sampled_num = args.sample_num
            list_clients = list(range(len(train_loaders)))
            sampled_index = random.sample(list_clients, sampled_num)
            epoch_start_time = time.time()
            for i in range(len(sampled_index)):
                sampled_loaders.append(train_loaders[sampled_index[i]])
            
            sampled_clients = list(zip(sampled_index, sampled_loaders))
            step_vals = algorithm.update(sampled_clients, args.local_epochs)
            checkpoint_vals['epoch_time'].append(time.time() - epoch_start_time)

            for key, val in step_vals.items():
                checkpoint_vals[key].append(val)

            if (epoch % checkpoint_freq == 0) or (epoch == args.global_rounds - 1):
                results = {
                    'epoch': epoch,
                }
                results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)
                for key, val in checkpoint_vals.items():
                    results[key] = np.mean(val)

                train_acc = 0
                evals = zip(eval_loader_names, eval_loaders, eval_weights)
                for name, loader, weights in evals:
                    acc = misc.accuracy(algorithm, loader, weights, args.device)
                    if name != 'test':
                        train_acc += acc
                    else:
                        test_acc = acc
                results['train_acc'] = train_acc / len(train_data)
                results[test_name] = test_acc

                if results['train_acc'] > best_train_acc:
                    save_checkpoint(f"model_{args.train_index}_{test_name}.pkl")
                    best_train_acc = results['train_acc']
                    best_test_acc = results[test_name]

                results_keys = list(results.keys())
                if results_keys != last_results_keys:
                    misc.print_row(results_keys, colwidth=12)
                    last_results_keys = results_keys
                misc.print_row([results[key] for key in results_keys],
                    colwidth=12)

                results.update({
                    'hparams': hparams,
                    'args': vars(args)
                })
                checkpoint_vals = collections.defaultdict(lambda: [])

        print("best_train_acc: {:.10f}".format(best_train_acc))
        print("best_test_acc: {:.10}".format(best_test_acc))

        result_dict["Test_env"]  = dataset.ENVIRONMENTS[args.test_envs[0]]
        result_dict["Train_acc"] = best_train_acc
        result_dict["Test_acc"] = best_test_acc
        with open(os.path.join(args.output_dir, args.dataset, args.algorithm, result_file), mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(result_dict.values())

        avg_acc += np.array([result_dict["Train_acc"], result_dict["Test_acc"]])

    result_dict["Test_env"] = "Avg"
    result_dict["Train_acc"] = avg_acc[0] / len(domains)
    result_dict["Test_acc"] = avg_acc[1] / len(domains)
    with open(os.path.join(args.output_dir, args.dataset, args.algorithm, result_file), mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(result_dict.values())

    with open(stdout_file, "a") as out_file, open(stderr_file, "a") as err_file:
        out_file.write('End\n')
        err_file.write('End\n')