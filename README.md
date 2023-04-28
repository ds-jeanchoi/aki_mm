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
python notext_model.py -run "SNUH" 
```

### Run ML model 
```
python cu_model.py -run "SNUH" 
```


### Results 
- selected best scores among 5 metrics based on f1-score
- 수정

|no|batch-size|learning-rate|precision|weight-deacy|metric|f1-score|AUROC|AUPRC|
|------|---|---|--|--|--|--|--|--|
|1|128|1.0e-4|16|1.0e-3|specificity|**0.507**|0.905|**0.501**|
|2|128|1.0e-4|16|1.0e-4|specificity|**0.509**|0.902|**0.498**|
|3|128|5.0e-4|16|1.0e-3|f1|0.489|0.891|0.456|
|4|128|5.0e-4|16|1.0e-4|auprc|0.480|0.891|0.406|
|5|128|1.0e-3|16|1.0e-3|auprc|0.426|0.866|0.328|
|6|56|1.0e-3|16|1.0e-3|auroc|0.440|0.890|0.429|
|7|56|1.0e-4|16|1.0e-3|auprc|0.507|0.902|0.495|
|8|28|1.0e-4|16|1.0e-3|auroc|0.499|0.896|0.447|
|9|28|5.0e-4|16|1.0e-3|f1|0.398|0.882|0.342|
|10|28|1.0e-3|16|1.0e-3|auroc|0.392|0.883|0.326|

