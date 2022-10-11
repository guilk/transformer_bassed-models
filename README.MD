## Transformer-based Object Detection and Segmentation Models
The code is based on Detectron2.
## Docker Installation

### Step 1: pull image docker
```bash
git pull guilk/gm:latest
```
### Step 2: launch docker image
```bash
docker run -it --rm --gpus all --ipc host  -v $CODE_DIR:/code -v $DATA_DIR:/data guilk/gm:latest  /bin/bash
```

## Model Evaluation

### Step 3: prepare datasets. 
Detectron2 has builtin support for a few datasets. The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`. Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  coco/
  lvis/
  cityscapes/
  VOC20{07,12}/
```
You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

Expected dataset structure for [COCO instance/keypoint detection](https://cocodataset.org/#download):
```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Step 4: Training
All configs can be trained with:

```
python tools/lazyconfig_train_net.py --config-file configs/path/to/config.py
```
By default, we use 64 GPUs with batch size as 64 for training.

### Step 5: Evaluation
Model evaluation can be done similarly:
```
python tools/lazyconfig_train_net.py --config-file configs/path/to/config.py --eval-only train.init_checkpoint=/path/to/model_checkpoint
```


