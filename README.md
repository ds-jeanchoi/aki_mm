# AKI prediction using multimodal data

The AKI multimodal prediction model works on python >= 3.8

### Install requirments.text
```
pip install -r requirements.txt
```


The AKI multimodal prediction model is built upon AUTOGLUON

### Install autogluon
> Requirements.text already includes installation. However, use can see details in website https://auto.gluon.ai/dev/index.html#installation

> and https://auto.gluon.ai/dev/tutorials/multimodal/index.html
```
pip install autogluon 
```
### Install cuML
> See datils in website https://github.com/rapidsai/cuml

### Run model
> Specify which data you are inferencing for : "SNUH", "KMC"  in -run


```
python model.py -run "SNUH"
```

### Hyperparameters
```
hyp =  {"model.names": ["hf_text", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
    "data.text.normalize_text": False,
    "data.categorical.convert_to_text": False,    
    "env.batch_size": 128,
    "optimization.learning_rate": 1.0e-3,
    "optimization.max_epochs": 20}
```

### Run model with no text
```
python notext_model.py --run "SNUH" 
```

### Run ML model 
```
python cu_model.py --run "SNUH" 
```


### Results 
- selected best scores among 5 metrics based on f1-score
- TBA

|no|batch-size|learning-rate|precision|weight-deacy|metric|f1-score|AUROC|AUPRC|
|------|---|---|--|--|--|--|--|--|
|1|128|1.0e-4|16|1.0e-3|specificity|** 00**|0.905|**00 **|
|2|128|1.0e-4|16|1.0e-4|f1|**0.485**|0.904|**0.488**|
|3|128|5.0e-4|16|1.0e-3|f1| |0.891||
|4|128|5.0e-4|16|1.0e-4|auprc| |0.891||
|5|128|1.0e-3|16|1.0e-3|auprc| |0.866||
|6|56|1.0e-3|16|1.0e-3|auroc| |0.890||
|7|56|1.0e-4|16|1.0e-3|auprc| |0.902||
|8|28|1.0e-4|16|1.0e-3|auroc| |0.896||
|9|28|5.0e-4|16|1.0e-3|f1| |0.882||
|10|28|1.0e-3|16|1.0e-3|auroc| |0.883||

