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
parser.add_argument("-task", "--task", dest="task", action="store")
args = parser.parse_args()



#SNUH input path
path  ="./new_input" 
task = args.task
#=args.input
#"./org_input/"
model_num = "m13"
output_path = './final_output_0428_1/'


run = args.run
#run = #KMC  SNUH  SNUH_test



#hyper parameter


#import params
# config_path = './hyp.json'
# with open(config_path) as f:
#     hyp  = json.load(f)

hyp =  {"model.names": ["hf_text", "categorical_mlp", "numerical_mlp", "fusion_mlp"],
    "data.text.normalize_text": False,
    "data.categorical.convert_to_text": False,
    "env.batch_size": 128,
    #"env.precision": 32,
    #"optimization.learning_rate": 1.0e-3,
    "optimization.weight_decay": 1.0e-4,
    "optimization.max_epochs": 20}

print(hyp)

os.makedirs(output_path, exist_ok=True)

with open(output_path+"hyperparams.json", "w") as json_file:
    json.dump(hyp, json_file)


metrics = {'f1' : 'f1' ,
           'auroc' :'roc_auc',
           'sensit' :'sensitivity', 
           'speci' : 'specificity',
           'AUPRC' :'average_precision'
        }

#input
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


if run in ('SNUH' ,  'SNUH_test') :

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
print("val",dev_df.columns)
print("test",test_df.columns)

print('Number of training samples:', len(train_df))
print('Number of dev samples:', len(dev_df))
print('Number of test samples:', len(test_df))
print('Number of calib  samples:', len(cal_df))


# Main model

for name, metric in metrics.items():
    #make folder if doens't exist
    save_path = output_path+str(run)+'/'+str(name)
    os.makedirs(save_path, exist_ok=True)
    
    tm = localtime()
    print(">>>>starting at: ", strftime('%Y-%m-%d %I:%M:%S %p', tm))
    start = time()

    path_model_name = 'src2/'+ model_num +name
    #textall+normal+batch56+conv
    print("[fitting",name, "start]")

#train MM

    if os.path.exists(path_model_name):
        predictor = MultiModalPredictor.load(path_model_name)
        print("file exists, load complete")
    else :
        print("--------no existing file found, new train start--------")
        predictor = MultiModalPredictor(label='new_total_aki', eval_metric=metric, path = path_model_name)  #alldata+metric
        predictor.fit(train_df, tuning_data = dev_df,  hyperparameters=hyp )
    print(predictor.fit_summary(verbosity=4, show_plot=True))

    print(name, "train complete, start evaluation")

#result
    performance = predictor.evaluate(test_df, metrics=['roc_auc','f1', 'recall', 'precision' ,'accuracy'])
    print(performance)

    eval_dict= {}
    label_col = 'new_total_aki'
    score_col = 'score'

    if task=='test' :
        
        df_groups = {
            'train': train_df,
            'test': dev_df }   #val set
        
        for df_name, df_group in df_groups.items():
        
            labels_df = df_group[label_col].values
            pred_df =predictor.predict(df_group[input_cols])

            proba_df  = predictor.predict_proba(df_group[input_cols]).iloc[:, 1]
            list = 'transformer' + '_' + df_name
            
            eval_dict[list] = pd.DataFrame({
                            label_col: labels_df,
                            score_col: proba_df,
                            'y_pred' : pred_df })

    elif task =='eval' :

#test data
#
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
            platt = PlattCalibrator(log_odds=True)
            platt.fit(proba_cal.values, labels_cal)
            
            isotonic = IsotonicRegression(out_of_bounds='clip',
                                      y_min=proba_cal.min(),
                                      y_max=proba_cal.max())
            isotonic.fit(proba_cal.values, labels_cal)
            
            # save model
            joblib.dump(platt, save_path+'/transformer_platt.pkl')
            joblib.dump(isotonic, save_path+'/transformer_isotonic.pkl')
        
        if run == "KMC" :
            platt = joblib.load(load_path+'/transformer_platt.pkl')
            isotonic = joblib.load(load_path+'/transformer_isotonic.pkl')
            
        platt_probs = platt.predict(proba_val.values)
        isotonic_probs = isotonic.predict(proba_val.values)
        # platt scale
        list =  'transformer' + '+platt'
        eval_dict[list] = pd.DataFrame({
                                label_col: labels_val,
                                score_col: platt_probs,
                                'y_pred' : pred_val })
        # isotonic      
        list =  'transformer' + '+isotonic'
        eval_dict[list] = pd.DataFrame({
                                label_col: labels_val,
                                score_col: isotonic_probs,
                                'y_pred' : pred_val })



    print("start printing results")
    n_bins = 15
    
#proba    
#calibrated probability for optimal threshold
    print(eval_dict.keys())


    if task  == 'test' :
        if os.path.isfile(save_path+'/auc.csv') :
            d = pd.read_csv(save_path+'/auc.csv')
        else :
            d = df_val['new_total_aki']
            
            
            
        for key, value in eval_dict.items() :
            if key.split("_")[-1] == 'test' :

                a  = pd.DataFrame(value['score'])
                d = pd.concat([d, a.rename(columns={'score':key})], axis =1)
    
    
    if task  == 'eval' :
        if os.path.isfile(save_path+'/auc.csv') :
            d = pd.read_csv(save_path+'/auc.csv')
        else :
            d = df_test['new_total_aki']

        for key, value in eval_dict.items() :
            a  = pd.DataFrame(value['score'])
            d = pd.concat([d, a.rename(columns={'score':key})], axis =1)
    
    d.to_csv(save_path+'/auc.csv', index=False)


    


#prediction

    if os.path.isfile(save_path+'/pred.csv') :
        d = pd.read_csv(save_path+'/pred.csv')
    else :
        d = df_val['new_total_aki']

    for key, value in eval_dict.items() :
        if key.split("_")[-1] == 'val' :
            a  = pd.DataFrame(value['y_pred'])
            d = pd.concat([d, a.rename(columns={'y_pred':key})], axis =1)
    d.to_csv(save_path+'/pred.csv', index=False)



    df_result = compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins, save_plot_path=save_path)


    print("--------",name, "done-------")

