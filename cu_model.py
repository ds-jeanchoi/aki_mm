import cupy as cp
import pandas as pd
from cuml.ensemble import RandomForestClassifier
from cuml.svm import SVC
from cuml import LogisticRegression
from xgboost import XGBClassifier
from sklearn.calibration import IsotonicRegression
import pickle
import joblib
from calibration_module.calibrator import  PlattCalibrator
from calibration_module.utils import compute_calibration_summary
from time import time, strftime, localtime
import json
import argparse
import os


#arugment
parser = argparse.ArgumentParser()
parser.add_argument("-run", "--run", dest="run", action="store")
parser.add_argument("-task", "--task", dest="task", action="store")
args = parser.parse_args()



task = args.task

output_path = './final_output_0428/'
if task == "eval" :
    config_path= './config.json'

elif task =="test" :
    config_path= './config_test.json'


path = "./new_input"
run = args.run

#KMC  SNUH  SNUH_test


#import params
with open(config_path) as f:
    param_dict = json.load(f)

#import data

df_train = pd.read_csv(path + "/train.csv")
df_val = pd.read_csv(path + "/val.csv")
df_cal = pd.read_csv(path + "/cal.csv")


drop = ['new_diagnosis','new_opname','is_ga' , 'multiple_within_7days_yes','Others']
df_train = df_train.drop(columns=drop)
df_val = df_val.drop(columns=drop)
df_cal = df_cal.drop(columns=drop)

if run in ('SNUH', 'SNUH_test') :

    df_test = pd.read_csv(path + "/test.csv")
    df_test = df_test.drop(columns=drop)

elif run == 'KMC' :

    drop = ['index','Others','sex','new_diagnosis','new_opname']
    df_test = pd.read_csv(path+"/kmc_test.csv")
    df_test = df_test.drop(columns = drop)

else :
    print("input run error")

 
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)
df_val = pd.DataFrame(df_val)
df_cal = pd.DataFrame(df_cal)


print("train:", df_train.shape)
print("train_cols:", df_train.columns)
print("test:", df_test.shape)
print("test_cols:", df_test.columns)
print("val:", df_val.shape)
print("val_cols:", df_val.columns)
print("cal:", df_cal.shape)
print("cal_cols:", df_cal.columns)


cudf_train   = cp.asnumpy(df_train)
cudf_test   = cp.asnumpy(df_test)
cudf_val   = cp.asnumpy(df_val)
cudf_cal   = cp.asnumpy(df_cal)


cols = df_train.columns
input_cols = cols.drop('new_total_aki')
label_col = 'new_total_aki'


print("data import success")




metrics = [
      'f1', 'auroc','sensit' , 'speci' ,  'AUPRC'
      
     ]

for i in range(len(metrics)) :
    #create folder if doesn't exist
    save_path = output_path+str(run)+'/'+str(metrics[i])
    os.makedirs(save_path, exist_ok=True)

    tm = localtime()
    print(">>>>starting at: ", strftime('%Y-%m-%d %I:%M:%S %p', tm))
    start = time()

    print("[fitting",metrics[i], "start]")
    xgb_params = param_dict[metrics[i]]['xgb']
    RF_params = param_dict[metrics[i]]['RF']
    LR_params = param_dict[metrics[i]]['LR']
    svc_params = param_dict[metrics[i]]['svc']

    print(svc_params)
    load_path = output_path+'SNUH/' + str(metrics[i])
    if run == "SNUH" :
        xgb = XGBClassifier(tree_method='gpu_hist', **xgb_params)
        xgb.fit(df_train[input_cols].values, df_train[label_col].values)
        joblib.dump(xgb, save_path+'/xgb.pkl')
        print("[xgb trained]")
        RF = RandomForestClassifier(**RF_params)
        RF.fit(cudf_train[:,:-1], cudf_train[:,-1])
        joblib.dump(RF, save_path+'/RF.pkl')
        print("[RF trained]")
        LR= LogisticRegression(**LR_params)
        LR.fit(cudf_train[:,:-1], cudf_train[:,-1])
        joblib.dump(LR, save_path+'/LR.pkl')
        print("[LR trained]")
        svc = SVC(probability=True, **svc_params)
        svc.fit(cudf_train[:,:-1], cudf_train[:,-1])
        joblib.dump(svc, save_path+'/svc.pkl')
        print("[svc trained]")
        print("----train finished and creating metrics---")
    end = time() 
    print(load_path)
    if run == "KMC" :
        xgb = joblib.load(load_path+'/xgb.pkl')
        RF  = joblib.load(load_path+'/RF.pkl')
        LR = joblib.load(load_path+'/LR.pkl')
        svc  = joblib.load(load_path+'/svc.pkl')
        print("----load finished and creating metrics---")
    
    
    
    
    
    print('>>time elapsed:', end - start)

    estimators = {
     'xgb': xgb
     ,
    'RF': RF
    ,  'LR': LR,
       'svc' : svc
        }
    if task=='test' :
        df_groups = {
            'train': df_train,
            'test': df_val
            }

        cudf_groups = {
             'train': cudf_train,
             'test': cudf_val
             }

    if task =='eval' :
        df_groups = {
            'val': df_test
    }

        cudf_groups = {
            'val': cudf_test
    }

    # Creating list
    list = []
    # Creating a dictionary
    eval_dict = {}
    label_col = 'new_total_aki'
    score_col = 'score'

    for name, estimator in estimators.items():
        if estimator == 'xgb' :
            for df_name, df_group in df_groups.items():
                list = name + '_' + df_name
                labels_val = df_group[label_col].values
                proba_val = estimator.predict_proba(df_group[input_cols].values)[:, 1]
                pred_val = estimator.predict(df_group[input_cols].values)

                eval_dict[list] = pd.DataFrame({
                    label_col: labels_val,
                    score_col: proba_val,
                    'y_pred': pred_val})

                if df_name == 'val':
                    # platt scaling for val data
                    labels_cal = df_cal[label_col].values
                    proba_cal = estimator.predict_proba(df_cal[input_cols].values)[:, 1]
                    #pred_test = estimator.predict(df_test[input_cols].values)

                    
                    if run == "SNUH" :

                        #fit platt
                        platt = PlattCalibrator(log_odds=True)
                        platt.fit(proba_cal, labels_cal)
                        #fit isotonic
                        isotonic = IsotonicRegression(out_of_bounds='clip',
                                                      y_min=proba_cal.min(),
                                                      y_max=proba_cal.max())
                        isotonic.fit(proba_cal.values, labels_cal)
                        #save model
                        joblib.dump(platt, save_path+'/'+ name +'_platt.pkl')
                        joblib.dump(isotonic, save_path+'/'+ name +'_isotonic.pkl')

                    if run == "KMC" :
                        #load model
                        platt = joblib.load(load_path+'/'+ name +'_platt.pkl')
                        isotonic = joblib.load(load_path+'/'+ name +'_isotonic.pkl')
                    
                    
                    platt_probs = platt.predict(proba_val)
                    isotonic_probs = isotonic.predict(proba_val.values)
                    

                    list = name + '_' + df_name + '+platt'
                    eval_dict[list] = pd.DataFrame({
                        label_col: labels_val,
                        score_col: platt_probs,
                        'y_pred': pred_val})
                    
                    list =  name  + '+isotonic'
                    eval_dict[list] = pd.DataFrame({
                                label_col: labels_val,
                                score_col: isotonic_probs,
                                'y_pred' : pred_val })

        else :
            for df_name, df_group in cudf_groups.items():
                list = name + '_' + df_name
                labels_val = df_group[:,-1]
                proba_val = estimator.predict_proba(df_group[:,:-1])[:, 1]
                pred_val = estimator.predict(df_group[:,:-1])

                eval_dict[list] = pd.DataFrame({
                    label_col: labels_val,
                    score_col: proba_val,
                    'y_pred': pred_val})

                if df_name == 'val':
                 # platt scaling for val data
                    list = name + '_' + df_name + '+platt'
                    labels_cal = cudf_cal[:, -1]
                    proba_cal  = estimator.predict_proba(cudf_cal[:, :-1])[:, 1]
                    #pred_test = estimator.predict(cudf_test[:,:-1])

                    if run == "SNUH" :

                        #fit platt
                        platt = PlattCalibrator(log_odds=True)
                        platt.fit(proba_cal, labels_cal)
                        #fit isotonic
                        isotonic = IsotonicRegression(out_of_bounds='clip',
                                                      y_min=proba_cal.min(),
                                                      y_max=proba_cal.max())
                        isotonic.fit(proba_cal, labels_cal)
                        #save model
                        joblib.dump(platt, save_path+'/'+ name +'_platt.pkl')
                        joblib.dump(isotonic, save_path+'/'+ name +'_isotonic.pkl')

                    if run == "KMC" :
                        #load model
                        platt = joblib.load(load_path+'/'+ name +'_platt.pkl')
                        isotonic = joblib.load(load_path+'/'+ name +'_isotonic.pkl')
                    
                    
                    platt_probs = platt.predict(proba_val)
                    isotonic_probs = isotonic.predict(proba_val)
                    

                    list = name + '_' + df_name + '+platt'
                    eval_dict[list] = pd.DataFrame({
                        label_col: labels_val,
                        score_col: platt_probs,
                        'y_pred': pred_val})
                    
                    list = name  + '+isotonic'
                    eval_dict[list] = pd.DataFrame({
                                label_col: labels_val,
                                score_col: isotonic_probs,
                                'y_pred' : pred_val })

    print(eval_dict.keys())


#proba
#calibrated probability for optimal threshold
    if task  == 'test' :
        if os.path.isfile(save_path+'/auc.csv') :
            d = pd.read_csv(save_path+'/auc.csv')
        else :
            d = df_test['new_total_aki']
        for key, value in eval_dict.items() :
            if key.split("_")[-1] == 'test' :
                a  = pd.DataFrame(value['score'])
                d = pd.concat([d, a.rename(columns={'score':key})], axis =1)
    
    
    if task == "eval":
        if os.path.isfile(save_path+'/auc.csv') :
            d = pd.read_csv(save_path+'/auc.csv')
        else :
            d = df_val['new_total_aki']
    
        for key, value in eval_dict.items() :

            #if key.split("_")[1] == 'val' :
            a = pd.DataFrame(value['score'])
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

    n_bins = 15
    
    #save_path = output_path + str(run) + '/' + str(metrics[i])
    df_result =compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins, save_plot_path=save_path)
