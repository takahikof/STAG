# Token Adaptation via Side Graph Convolution for Temporally and Spatially Efficient Fine-tuning of 3D Point Cloud Transformers 🦌
Takahiko Furuya (University of Yamanashi)

Currently under review. Code will be released after being accepted.

# Abstract
While existing parameter-efficient fine-tuning (PEFT) of 3D point cloud Transformers attempt to minimize the number of tunable parameters, they still suffer from high temporal and spatial computational costs during fine-tuning. This paper proposes a novel PEFT algorithm called Side Token Adaptation on a neighborhood Graph (STAG) to improve temporal and spatial efficiency. This paper also proposes a benchmark called PCC13, which consists of 13 publicly available labeled 3D point cloud datasets, to enable comprehensive evaluation with diverse point cloud data.
| Algorithms | # of tunable parameters | Time per epoch [s] | VRAM consumption [GB] | Overall accuracy [%] on PCC13 |
|:-------|--------:|--------:|--------:|--------:|
| Full fine-tuning | 22.09M | 4.29 | 6.1| 80.8 |
| IDPT | 1.70M | 5.43 | 6.6 | 80.2 |
| DAPT | 1.09M | 3.57 | 4.7 | 80.6 |
| Point-PEFT | 0.77M | 13.66 | 13.2 | 80.3 |
| PPT | 1.04M | 9.42 | 13.5 | 80.7 |
| PointGST | 0.62M | 5.59 | 3.6 | 80.7 |
| STAG-std (standard size) | 0.43M | 2.59 | 2.0 | 81.2 |
| STAG-sl (slightly large size) | 1.02M | 3.10 | 3.0 | 81.7 |
* Pre-trained model: Point-MAE, batch size: 32, hardware configuration: Intel Core i7-14700KF CPU and NVIDIA RTX 6000 Ada GPU

# Pre-requisites
My code has been tested on Ubuntu 22.04. I highly recommend using the Docker container "nvcr.io/nvidia/pytorch:23.10-py3", which is provided by [Nvidia NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).
After launching the Docker container, run the following shell script to install the prerequisite libraries.
```
cd prepare
./prepare.sh
```
## Datasets
See [DATASET.md](DATASET.md) for details.
## Pre-trained parameters
See [PARAM.md](PARAM.md) for details.

# Fine-tuning evaluation using PCC13 benchmark
Once the dataset and pre-trained parameters are fully prepared, you can proceed to the fine-tuning experiments.

## [Baseline] Full fine-tuning
Full fine-tuning, which finetunes all parameters within the DNN, serves as the baseline method.
```
# Point-MAE
cd FullFinetuning/Point-MAE/
./Run_finetune.sh
```
```
# MaskLRF
cd FullFinetuning/MaskLRF/
./Run_finetune.sh
```
```
# Uni3D-S
cd FullFinetuning/Uni3D/
./Run_finetune.sh
```

## [Proposed] Efficient fine-tuning by STAG
Depending on the script file you run, you can choose STAG-std or STAG-sl.
```
# Point-MAE
cd STAG/Point-MAE/
./Run_finetune_STAG-std.sh
./Run_finetune_STAG-sl.sh
```
```
# MaskLRF
cd STAG/MaskLRF/
./Run_finetune_STAG-std.sh
./Run_finetune_STAG-sl.sh
```
```
# Uni3D-S
cd STAG/Uni3D/
./Run_finetune_STAG-std.sh
./Run_finetune_STAG-sl.sh
```
