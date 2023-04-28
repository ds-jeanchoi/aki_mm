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

### Run model with no text
```
python notext_model.py -run "SNUH" -input "./input"
```

### Run ML model 
```
python cu_model.py -run "SNUH" -input "./input"
```


|제목|내용|설명|
|------|---|---|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
|테스트1|테스트2|테스트3|
