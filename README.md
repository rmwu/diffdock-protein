# DiffDock modified for protein-protein

Note: currently I only have atom-level information. I'm (quickly) adding back the residue-level ESM embeddings so should have an update today. Of course you should feel free to read through / poke around.

## Model training:

```
./src/train.sh
```

To modify dataset, change configuration file.


## Model inference:

```
./src/predict.sh
```

Currently the model only works for >1 GPU :') But trust me it'll take much too long if you try with only 1.

DIPS and DB5 work. SabDab has not been updated for a long time.

## Data

My data splits

```
ln -s /data/rsg/chemistry/rmwu/src/sandbox/glue/data data
```

Note: you can look at the above directory for clues as to formatting (mainly, instead of Octavian's 3 files, 1 per split, I have a CSV with the split = `train` `val` `test` appended as a column.

You can specify the data's location via `data_path` (you are welcome to use the existing path to my directory)

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
