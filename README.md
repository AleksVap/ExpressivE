# ExpressivE

This repository contains the official source code for the ExpressivE model, presented at ICLR 2023 in our [paper](https://openreview.net/forum?id=xkev3_np08z) "ExpressivE: A Spatio-Functional Embedding For Knowledge Graph Completion".
The repository includes the following:

1. the implementation of ExpressivE
2. the code for training and testing ExpressivE on WN18RR and FB15k-237 to reproduce the results presented in our paper
3. an environment.yml to set up a conda environment with all dependencies automatically

# Requirements 

* Python >= 3.9
* PyKEEN 1.7.0
* Pytorch 1.10.2
* Tensorboard 2.8.0

# Installation

We have provided an `environment.yml` file that can be used to create a conda environment with all required dependencies. 
To install ExpressivE, simply run `conda env create -f environment.yml` to create the conda 
environment `Env_ExpressivE`. Afterward, use `conda activate Env_ExpressivE` to activate the environment before rerunning our experiments.

# Running ExpressivE

Training and evaluation of ExpressivE are done by running the `run_experiments.py` file. In particular, a configuration file
must be specified for an ExpressivE model, containing all model, training, and evaluation parameters. The best
configuration files for WN18RR and FB15k-237 are provided in the `Best_Configurations` directory and can be adapted to
try out different parameter configurations. To run an experiment, the following parameters need to be specified:

- `config` contains the path to the model configuration (e.g., `config=Best_Configurations/ExpressivE/Base_ExpressivE/WN18RR.json`)
- `train` contains `true` if the model shall be trained and `false` otherwise.
- `test` contains `true` if the model shall be evaluated on the test set and `false` otherwise.
- `expName` contains the name of the experiment (e.g., `expName=Base_ExpressivE_WN18RR`)
- `gpu` contains the id of the gpu that shall be used (e.g., `gpu=0`)
- `seeds` contains the seeds for repeated runs of the experiment (e.g., `seeds=1,2,3`)

Finally, you can run an experiment
with `python run_experiments.py config=<config> train=<true|false> test=<true|false> expName=<expName> gpu=<gpuID> seeds=<seeds>`,
where angle brackets represent a parameter value. If the `test` parameter is set to `true`, then the selected model
will be evaluated on the test set, and the evaluation results will be stored in `Benchmarking/final_result/<expName>`.

# Reproducing the Results

In the following, we provide the commands to reproduce ExpressivE's benchmark results (listed in Table 3 of our [paper](https://openreview.net/forum?id=xkev3_np08z)):

## Benchmarks

### WN18RR

Train & Test: `python run_experiments.py gpu=0 train=true test=true seeds=1,2,3 config=Best_Configurations/ExpressivE/Base_ExpressivE/WN18RR.json expName=Base_ExpressivE_WN18RR`

### Fb15k-237

Train & Test: `python run_experiments.py gpu=0 train=true test=true seeds=1,2,3 config=Best_Configurations/ExpressivE/Functional_ExpressivE/FB15k-237.json expName=Functional_ExpressivE_FB15k-237`

## Per-Relation Evaluation

To evaluate the performance of the Base ExpressivE model per relation on WN18RR, first, train a Base ExpressivE model as described above. 
After the training has finished, run the following commands (which evaluate the trained models of each seed):

Seed 1: `python run_per_relation_evaluation.py model_dir=Benchmarking sub_dir=Base_ExpressivE_WN18RR_seed_1 config_path=Best_Configurations/ExpressivE/Base_ExpressivE/WN18RR.json`

Seed 2: `python run_per_relation_evaluation.py model_dir=Benchmarking sub_dir=Base_ExpressivE_WN18RR_seed_2 config_path=Best_Configurations/ExpressivE/Base_ExpressivE/WN18RR.json`

Seed 3: `python run_per_relation_evaluation.py model_dir=Benchmarking sub_dir=Base_ExpressivE_WN18RR_seed_3 config_path=Best_Configurations/ExpressivE/Base_ExpressivE/WN18RR.json`

The relation-wise performance results will be stored at `Benchmarking/per_relation_result/Base_ExpressivE_WN18RR_seed_<s>`, where `<s>` is `1`, `2`, or `3`.

# Citation 

If you use this code or its corresponding [paper](https://openreview.net/forum?id=xkev3_np08z), please cite our work as follows:

```
@inproceedings{
pavlovic2023expressive,
title={ExpressivE: A Spatio-Functional Embedding For Knowledge Graph Completion},
author={Aleksandar Pavlovi{\'c} and Emanuel Sallinger},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=xkev3_np08z}
}
```

# Contact

Aleksandar Pavlović

Research Unit of Databases and Artificial Intelligence

Vienna University of Technology (TU Wien)

Vienna, Austria

<aleksandar.pavlovic@tuwien.ac.at>

# Licenses

The benchmark datasets WN18RR and FB15k-237 are already included in the PyKEEN library. PyKEEN uses the MIT license.
FB15k-237 is a subset of FB15k which uses the CC BY 2.5 license. The license of FB15k-237 and WN18RR is unknown.
This project runs under the MIT license.

Copyright (c) 2023 Aleksandar Pavlović
