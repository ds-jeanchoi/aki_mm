# aki_mm

The AKI multimodal prediction model works on python >= 3.8

### Install requirments.text
```
pip install -r requirements.txt
```


The AKI multimodal prediction model is built upon AUTOGLUON

### Install autogluon
Use can see details in website 
https://auto.gluon.ai/dev/index.html#installation
https://auto.gluon.ai/dev/tutorials/multimodal/index.html
```
pip install autogluon 
```

### Run model
> Specify if your are running for SNUH/SNUH_test/KMC data in -run
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
