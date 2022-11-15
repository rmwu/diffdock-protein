# DiffDock modified for protein-protein

## Model training:

```
./src/train.sh
```

To modify dataset, change configuration file.


## Model inference:

```
./src/predict.sh
```

Currently the model only works for >1 GPU :')

DIPS and DB5 work. SabDab has not been updated for a long time.

## Conda environment

```
pip install numpy

pip install dill
pip install tqdm
pip install pyyaml
pip install pandas

pip install scikit-learn
pip install biopython

# install PyTorch

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

pip install --upgrade e3nn

pip install tensorboard
pip install tensorboardX

# install compatible pytorch geometric in this order WITH versions

pip install --no-cache-dir  torch-scatter==2.0.6 torch-sparse==0.6.9 torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html

```
