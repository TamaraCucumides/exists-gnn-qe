## Installation ##
First install torch and cuda
```conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia```

Then install both torchcluster and torchscatter
```pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html```

Finally install torchdrug
```pip install torchdrug```

And yaml and easydict
```pip install easydict pyyaml```


## How to run? 
The folder script has the files to run, and useful configurations can be found in configuration folder. For example, for training on dataset FB15k-237, run 
``` python script/run.py -c config/fb15k237-train.yaml --gpus [0]```
