import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from calibration_module.utils import compute_calibration_summary
from calibration_module.calibrator import  PlattCalibrator
from sklearn.calibration import IsotonicRegression
import pickle
import joblib
from time import time, strftime, localtime
import os
import json
import argparse


#arguments
parser = argparse.ArgumentParser()
parser.add_argument("-run", "--run", dest="run", action="store")
args = parser.parse_args()


#input path
path  ="./new_input" 
#output path
output_path = './output/'



#specify KMC or SNUH
run = args.run
 


#hyperparams

hyp =  {"model.names": ["hf_text", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
        "data.text.normalize_text": False,
        "data.categorical.convert_to_text": False,
        'env.batch_size': 128, 
        "optimization.learning_rate": 1.0e-3,
        'optimization.optim_type':"adamw",
        'optimization.weight_decay': 0.0001,
        'optimization.max_epochs': 20}

print(hyp)

os.makedirs(output_path, exist_ok=True)

#save hyperparms as json
with open(output_path+"hyperparams.json", "w") as json_file:
    json.dump(hyp, json_file)


# evalulation metrics    
metrics = {'f1' : 'f1' ,
           'auroc' :'roc_auc',
           'sensit' :'sensitivity', 
           'speci' : 'specificity',
           'AUPRC' :'average_precision'
        }

#############
####input####
#############

df_train = pd.read_csv(path + "/train.csv")
df_val = pd.read_csv(path + "/val.csv")
df_cal = pd.read_csv(path+ "/cal.csv")


drop = ['is_ga', 'multiple_within_7days_yes', 'Others']
df_train = df_train.drop(columns=drop)
df_val = df_val.drop(columns=drop)
df_cal = df_cal.drop(columns =drop)

cols = df_train.columns
input_cols = cols.drop('new_total_aki')
label_col = 'new_total_aki'


if run == 'SNUH' :

    df_test = pd.read_csv(path + "/test.csv")
    df_test= df_test.drop(columns=drop)

elif run == 'KMC' :

    drop = ['Others','sex']
    df_test = pd.read_csv(path + "/kmc_test.csv")
    df_test = df_test.drop(columns = drop)

else :
    print("input run error")


feature_columns = df_train.columns
label = 'new_total_aki'
print(feature_columns)


train_df = df_train
dev_df = df_val[feature_columns]
test_df = df_test[feature_columns]
cal_df = df_cal[feature_columns]

print("train",train_df.columns)


print('Number of training samples:', len(train_df))
print('Number of dev samples:', len(dev_df))
print('Number of test samples:', len(test_df))
print('Number of calib  samples:', len(cal_df))


#############
# Main model#
#############


for name, metric in metrics.items():
    #make folder if doens't exist
    save_path = output_path+str(run)+'/'+str(name)
    os.makedirs(save_path, exist_ok=True)
    
    tm = localtime()
    print(">>>>starting at: ", strftime('%Y-%m-%d %I:%M:%S %p', tm))
    start = time()
    
    # save model name as
    path_model_name = save_path + "/model/" + name
 
    print("--------[fitting",name, "start]--------")

#train MM

    if os.path.exists(path_model_name):
        predictor = MultiModalPredictor.load(path_model_name)
        print("--------saved model exists, load complete--------")
    else :
        print("--------no existing file found, new train start--------")
        predictor = MultiModalPredictor(label='new_total_aki', eval_metric=metric, path = path_model_name)  #alldata+metric
        predictor.fit(train_df, tuning_data = dev_df,  hyperparameters=hyp )
    print(predictor.fit_summary(verbosity=4, show_plot=True))

    print(name, "--------train complete, start evaluation--------")

    
#predict test set
    performance = predictor.evaluate(test_df, metrics=['roc_auc','f1', 'recall', 'precision' ,'accuracy'])
    print(performance)

    eval_dict= {}
    label_col = 'new_total_aki'
    score_col = 'score'


    labels_val = test_df[label_col].values
    pred_val =predictor.predict(test_df[input_cols])
    proba_val  = predictor.predict_proba(test_df[input_cols]).iloc[:, 1]

    list = 'transformer' + '_val'

    eval_dict[list] = pd.DataFrame({
                        label_col: labels_val,
                        score_col: proba_val,
                        'y_pred' : pred_val })
    
#platt scaling
#fit with cal set, predict with test

    load_path = output_path+'SNUH/'+str(name)

    labels_cal = cal_df[label_col].values
    proba_cal = predictor.predict(cal_df[input_cols])


    if run == "SNUH" :
        #platt scale
        if os.path.isfile(save_path+'model/transformer_platt.pkl') :
            platt = joblib.load(save_path+'model/transformer_platt.pkl')           
        else :
            platt = PlattCalibrator(log_odds=True)
            platt.fit(proba_cal.values, labels_cal)
            joblib.dump(platt, save_path+'/transformer_platt.pkl')
            
        # isotonic    
        if os.path.isfile(save_path+'/transformer_isotonic.pkl') :
            isotonic = joblib.load(save_path+'/transformer_isotonic.pkl')
        else :
            isotonic = IsotonicRegression(out_of_bounds='clip',
                                      y_min=proba_cal.min(),
                                      y_max=proba_cal.max())
            isotonic.fit(proba_cal.values, labels_cal)
            joblib.dump(isotonic, save_path+'/transformer_isotonic.pkl')

    if run == "KMC" :
        platt = joblib.load(load_path+'/transformer_platt.pkl')
        isotonic = joblib.load(load_path+'/transformer_isotonic.pkl')

    platt_probs = platt.predict(proba_val.values)
    isotonic_probs = isotonic.predict(proba_val.values)
    
   
    list =  'transformer' + '+platt'
    eval_dict[list] = pd.DataFrame({
                            label_col: labels_val,
                            score_col: platt_probs,
                            'y_pred' : pred_val })
    
    list =  'transformer' + '+isotonic'
    eval_dict[list] = pd.DataFrame({
                            label_col: labels_val,
                            score_col: isotonic_probs,
                            'y_pred' : pred_val })



    print("--------start printing results--------")
    n_bins = 15
    


    print(eval_dict.keys())
    
    
#save probability for model comparison & caculate new threholds


    if os.path.isfile(save_path+'/auc.csv') :
        d = pd.read_csv(save_path+'/auc.csv')
    else :
        d = df_test['new_total_aki']

    for key, value in eval_dict.items() :
        a  = pd.DataFrame(value['score'])
        d = pd.concat([d, a.rename(columns={'score':key})], axis =1)

    d.to_csv(save_path+'/auc.csv', index=False)


    
#save prediction
    if os.path.isfile(save_path+'/pred.csv') :
        d = pd.read_csv(save_path+'/pred.csv')
    else :
        d = df_test['new_total_aki']

    for key, value in eval_dict.items() :
        if key.split("_")[-1] == 'val' :
            a  = pd.DataFrame(value['y_pred'])
            d = pd.concat([d, a.rename(columns={'y_pred':key})], axis =1)
    d.to_csv(save_path+'/pred.csv', index=False)



#create metrics.csv and graphs
    df_result = compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins, save_plot_path=save_path)


    print("--------",name, "done-------")

