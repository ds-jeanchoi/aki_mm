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
args = parser.parse_args()


#specify KMC or SNUH 
run = args.run

#hyperparms path
config_path= './config.json'
#input path
path = "./new_input"
#output path
output_path = './output/'

#import hyperparams
with open(config_path) as f:
    param_dict = json.load(f)


#############
####input####
#############
df_train = pd.read_csv(path + "/train.csv")
df_val = pd.read_csv(path + "/val.csv")
df_cal = pd.read_csv(path + "/cal.csv")


drop = ['new_diagnosis','new_opname','is_ga' , 'multiple_within_7days_yes','Others']
df_train = df_train.drop(columns=drop)
df_val = df_val.drop(columns=drop)
df_cal = df_cal.drop(columns=drop)

if run =='SNUH' :

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


print("----data import success----")



# evaluation metics
metrics = [
      'f1', 'auroc','sensit' , 'speci' ,  'AUPRC'
       ]


#############
# Main model#
#############

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
        
        os.makedirs(save_path+'/model', exist_ok=True)
        
        if os.path.isfile(load_path+'/model/xgb.pkl') :
            xgb = joblib.load(load_path+'/model/xgb.pkl')
        else :    
            xgb = XGBClassifier(tree_method='gpu_hist', **xgb_params)
            xgb.fit(df_train[input_cols].values, df_train[label_col].values)
            joblib.dump(xgb, save_path+'/model/xgb.pkl')
            print("[xgb trained]")
            
        if os.path.isfile(load_path+'/model/RF.pkl') :
            RF  = joblib.load(load_path+'/model/RF.pkl')         
        else :
            RF = RandomForestClassifier(**RF_params)
            RF.fit(cudf_train[:,:-1], cudf_train[:,-1])
            joblib.dump(RF, save_path+'/model/RF.pkl')
            print("[RF trained]")    
            
        if os.path.isfile(load_path+'/model/LR.pkl') :    
            LR = joblib.load(load_path+'/model/LR.pkl')          
        else :
            LR= LogisticRegression(**LR_params)
            LR.fit(cudf_train[:,:-1], cudf_train[:,-1])
            joblib.dump(LR, save_path+'/model/LR.pkl')
            print("[LR trained]") 
        
        if os.path.isfile(load_path+'/model/svc.pkl') :   
            svc  = joblib.load(load_path+'/model/svc.pkl')
        else :
            svc = SVC(probability=True, **svc_params)
            svc.fit(cudf_train[:,:-1], cudf_train[:,-1])
            joblib.dump(svc, save_path+'/model/svc.pkl')
            print("[svc trained]")
    print("----train finished and creating metrics---")
            
    end = time() 
    
    if run == "KMC" :
        xgb = joblib.load(load_path+'/model/xgb.pkl')
        RF  = joblib.load(load_path+'/model/RF.pkl')
        LR = joblib.load(load_path+'/model/LR.pkl')
        svc  = joblib.load(load_path+'/model/svc.pkl')
        print("----load finished and creating metrics---")
    

    
    print('>>time elapsed:', end - start)

    estimators = {
     'xgb': xgb,
     'RF': RF,  
     'LR': LR,
     'svc' : svc
        }


    df_groups = {'val': df_test}
    cudf_groups = {'val': cudf_test}

    list = []
    
    # Creating a dictionary for result
    eval_dict = {}
    label_col = 'new_total_aki'
    score_col = 'score'

    for name, estimator in estimators.items():
        
        #xgb model
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
                        if os.path.isfile(save_path+'/model/'+ name +'_platt.pkl') :
                            platt = joblib.load(load_path+'/model/'+ name +'_platt.pkl')                   
                        else :                    
                            platt = PlattCalibrator(log_odds=True)
                            platt.fit(proba_cal, labels_cal)   
                            joblib.dump(platt, save_path+'/model/'+ name +'_platt.pkl')
                            
                        if os.path.isfile(save_path+'/model/'+ name +'_isotonic.pkl') :                         
                            isotonic = joblib.load(load_path+'/model/'+ name +'_isotonic.pkl')
                        else :
                            #fit isotonic
                            isotonic = IsotonicRegression(out_of_bounds='clip',
                                                          y_min=proba_cal.min(),
                                                          y_max=proba_cal.max())
                            isotonic.fit(proba_cal.values, labels_cal)  
                            joblib.dump(isotonic, save_path+'/model/'+ name +'_isotonic.pkl')

                    if run == "KMC" :
                        #load model
                        platt = joblib.load(load_path+'/model/'+ name +'_platt.pkl')
                        isotonic = joblib.load(load_path+'/model/'+ name +'_isotonic.pkl')
                    
                    # calibrated prediction
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
        # LR, SVC, RF models
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
                 # platt scaling 
                    list = name + '_' + df_name + '+platt'
                    labels_cal = cudf_cal[:, -1]
                    proba_cal  = estimator.predict_proba(cudf_cal[:, :-1])[:, 1]
             

                    if run == "SNUH" :
                        if os.path.isfile(save_path+'/model/'+ name +'_platt.pkl') :
                            platt = joblib.load(load_path+'/model/'+ name +'_platt.pkl')                   
                        else :                    
                            platt = PlattCalibrator(log_odds=True)
                            platt.fit(proba_cal, labels_cal)   
                            joblib.dump(platt, save_path+'/model/'+ name +'_platt.pkl')
                            
                        if os.path.isfile(save_path+'/model/'+ name +'_isotonic.pkl') :                         
                            isotonic = joblib.load(load_path+'/model/'+ name +'_isotonic.pkl')
                        else :
                            #fit isotonic
                            isotonic = IsotonicRegression(out_of_bounds='clip',
                                                          y_min=proba_cal.min(),
                                                          y_max=proba_cal.max())
                            isotonic.fit(proba_cal.values, labels_cal)  
                            joblib.dump(isotonic, save_path+'/model/'+ name +'_isotonic.pkl')
                        
                            

                    if run == "KMC" :
                        #load model
                        platt = joblib.load(load_path+'/model/'+ name +'_platt.pkl')
                        isotonic = joblib.load(load_path+'/model/'+ name +'_isotonic.pkl')
                    
                    
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


#save probability for model comparison & caculate new threholds

    if os.path.isfile(save_path+'/auc.csv') :
        d = pd.read_csv(save_path+'/auc.csv')
    else :
        d = df_test['new_total_aki']

    for key, value in eval_dict.items() :
        a = pd.DataFrame(value['score'])
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

    n_bins = 15
   
#create metrics.csv and graphs
    #save_path = output_path + str(run) + '/' + str(metrics[i])
    df_result =compute_calibration_summary(eval_dict, label_col, score_col, n_bins=n_bins, save_plot_path=save_path)
