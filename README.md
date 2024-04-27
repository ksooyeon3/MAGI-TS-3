# MAGI for Time Series Forecasting

This repository provides the PyTorch implementation of the MAGI approach for multivariate time series forecasting where there exists some unknown underlying dynamic between the time series. 

Here is the directory tree for the key component of the repo
```
.
+-- old_experiment/
+-- data/
+-- scripts/
|   +-- magix/
+-- magix_final.ipynb
```

### Updated MAGI-X
Please see ``magix_final.ipynb`` for details of running the code. All source codes are stored under ``scripts/magix/``. 

### Old Experiment 
Here is the directory tree of the folder
```
.
+-- data/
|   +-- fn/
|   +-- hes1/
|   +-- lv/
+-- scripts/
|   +-- experiment.py
|   +-- utils.py
|   +-- magix/
|   +-- npode/
|   +-- torchdiffeq/
+-- params/
|   +-- fn.config
|   +-- hes1.config
|   +-- lv.config
```

For the old experiment running Neural ODE (``scripts/torchdiffeq/``) and NPODE (``scripts/npode/``). Please see ``scripts/experiment.py`` for the detail code to run each method. 

For running the comparison directly via ``scripts/experiment.py`` (please update ``scripts/magix`` first if we also want to run MAGI-X), we need to provide the path to the parameter config file and specify the path to the result directory, e.g., 

```sh
python3 scripts/experiment.py -p params/fn.config -r results/fn/01/
```

Here is one sample parameter config file:

```
data:example=fn
data:train=1
data:noise=0.1,0.1

experiment:no_run=100

magix:run=yes
magix:no_iteration=2500
magix:hidden_node=512

npode:run=yes
npode:no_iteration=500

nrode:run=yes
nrode:no_iteration=500
nrode:hidden_node=512
```

We first need to specify the system we want to run: ``fn/lv/hes1``. Next, we provide the setting for the noisy observations: fullly(1)/partial(2) observed data for "data:train" option and noise level for "data:noise" option. Last, we provide algorithm specific setting such as number of iterations. Note "nrode" stands for Neural ODE in the config file.

The groundtruth data are computed via default numerical integration provided in the ``scipy`` package. The outputs are stored in ``data/`` along with ``train1.txt`` and ``train2.txt`` where ``train1.txt`` contains observation index for the fully observed scenario and ``train2.txt`` contains observation index for the partially observed scenario. Here is the ``train1.txt`` file (fully observed example) for FN system:

```
0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160
0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160
```

The index starts with 0 since Python starts with index 0, but in the paper, we start with index 1. We can see at any particular time points, we have data for both components. Here is the ``train2.txt`` file (partially observed example) for FN system:

```
0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160
2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74,78,82,86,90,94,98,102,106,110,114,118,122,126,130,134,138,142,146,150,154,158
```

We can see that at any particular time point, we only observe one of the component. 