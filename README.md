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
> Specify which data you are running for : "SNUH", "SNUH_test", "KMC"  in -run

> Specify input file path in -input 
```
python model.py -run "SNUH" -input "./input"
```

### Hyperparameters
hyp =  {"model.names": ["hf_text", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
    "data.text.normalize_text": False,
    "data.categorical.convert_to_text": False,
    "env.batch_size": 56,
    "optimization.learning_rate": 1.0e-3,
    "optimization.max_epochs": 20}

### Run model with no text
```
python notext_model.py -run "SNUH" -input "./input"
```

### Run ML model 
```
python cu_model.py -run "SNUH" -input "./input"
```


### Results 

|no|batch-size|learning-rate|precision|metric|f1-score|AUPRC|
|------|---|---|--|--|--|--|
|1|테스트2|테스트3|ㅇㅇ|ㅇㅇ|--|--|--|
|2|테스트2|테스트3|ㅇㅇ|ㅇㅇ|--|--|--|
|3|테스트2|테스트3|ㅇㅇ|ㅇㅇ|--|--|--|
