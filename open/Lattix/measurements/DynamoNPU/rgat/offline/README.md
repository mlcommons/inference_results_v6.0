# R-GAT on Dynamo NPU - Offline Accuracy Run

## Setup
- Dataset: IGBH-tiny (256 validation samples, nodes 60000-60255)
- Framework: PyTorch + custom Dynamo NPU backend
- LoadGen version: 6.0.13
- Mode: AccuracyOnly, Offline scenario

## Run command
python main.py --scenario Offline --mode AccuracyOnly
