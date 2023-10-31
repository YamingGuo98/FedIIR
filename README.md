# Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships

Official PyTorch implementation for the ICML 2023 paper [Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships](https://proceedings.mlr.press/v202/guo23b.html). Our code structure is founded on [DomainBed](https://github.com/facebookresearch/DomainBed), an out-of-distribution generalization benchmark designed for the centralized scenario, but has been refactored to suit the federated learning.

## Requirements

This codebase is written in `Python3`, specifically developed using `Python 3.8.12`. We utilize PyTorch version `1.10.0` and CUDA version `10.2`. To install the necessary Python packages, you can use:

```
pip install -r requirements.txt
```

## How to Run

Firstly, download the dataset, such as RotatedMNIST:

```bash
python domainbed/download.py --data_dir=./data --dataset RotatedMNIST
```

Next, train the model. For instance, to execute the FedIIR algorithm on the RotatedMNIST dataset:

```bash
python main.py --dataset RotatedMNIST --algorithm FedIIR --num_clients 50 --output_dir train_output
```

The results will be saved in `train_output/RotatedMNIST/FedIIR`, including model files `*.pkl`, the result file `*.csv`, the standard output file `*_stdout`, and the standard error file `*_stderr`. It is worth noting that the main file `main.py` conducts experiments by  treating each domain within the dataset in turn as a test domain once.

The main file `main.py` also allows overloading the experiment configuration using parser arguments, some of the important ones are as follows:

* `--num_clients`: the number of total clients
* `--sample_num`: the number of sampled clients in one communication round
* `--global_epochs `: the number of global communication rounds
* `--local_epochs`: the number of local update epochs
* `--holdout_fraction`: the fraction of each domain set holdout for validation
* `--trial_seed`: the trial number (used for seeding split dataset)
* `--seed`: the seed for everything else

## Reference Github

* [https://github.com/facebookresearch/DomainBed](https://github.com/facebookresearch/DomainBed)
* [https://github.com/inouye-lab/FedDG_Benchmark](https://github.com/inouye-lab/FedDG_Benchmark)
* [https://github.com/atuannguyen/FedSR](https://github.com/atuannguyen/FedSR)

## Citation

Please consider citing our paper as

```
@InProceedings{pmlr-v202-guo23b,
  title = 	 {Out-of-Distribution Generalization of Federated Learning via Implicit Invariant Relationships},
  author =       {Guo, Yaming and Guo, Kai and Cao, Xiaofeng and Wu, Tieru and Chang, Yi},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  year = 	 {2023}
}

```
