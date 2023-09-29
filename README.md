## Installation ##
First install torch and cuda
```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then install both torchcluster and torchscatter
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
```

Finally install torchdrug
```
pip install torchdrug
```

And yaml and easydict
```
pip install easydict pyyaml
```


## How to run? 
The folder script has the files to run, and useful configurations can be found in configuration folder. For example, for training on dataset FB15k-237, run 
```
python script/run.py -c config/fb15k237-train.yaml --gpus [0]
```

## Datasets
* The original BetaE datasets are automatically downloaded when you run the code.
* We provide unanchored queries in this link.
* Test set of cyclic queries and unravelings are in the following link. Triangles come with 3 and 4 unravelings, labeled under id 3pi and 4pi. Squares are labeled respectively with labels 6pi and 8pi.
* In order to run/train with our queries, you must either replace the corresponding files in the kg-dataset folder, or create a new folder inside kg-dataset. For example: your-path/kg-datasets/cyclic-queries/N-betae/ where N can be FB15k-237, FB15k, or NELL. Remember to adjust the config files (.yaml) with this new path
