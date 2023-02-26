import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from calibration_module.utils import compute_calibration_summary
from calibration_module.calibrator import  PlattCalibrator
from time import time, strftime, localtime
import os
import argparse


#argument
parser = argparse.ArgumentParser()
parser.add_argument("-run", "--run", dest="run", action="store")
parser.add_argument("-input", "--input", dest="input", action="store")
args = parser.parse_args()




output_path = './final_output0224/'

path = args.input
run = args.run
#run = KMC  SNUH  SNUH_test

hyp = {"model.names": ["categorical_mlp", "numerical_mlp", "fusion_mlp"],
    "data.text.normalize_text": False,
    "data.categorical.convert_to_text": False,
    "optimization.max_epochs": 20}

print(hyp)


metrics = {
        'f1' : 'f1'
    , 'auroc' :'roc_auc'
    ,'sensit' :'sensitivity' 
    , 'speci' : 'specificity'
        }

#input
df_train = pd.read_csv(path + "/train.csv")
df_test = pd.read_csv(path + "/test.csv")


drop = ['is_ga', 'new_opname','new_diagnosis',  'multiple_within_7days_yes', 'Others']
df_train = df_train.drop(columns=drop)
df_test = df_test.drop(columns=drop)

cols = df_train.columns
input_cols = cols.drop('new_total_aki')
label_col = 'new_total_aki'


if run in ('SNUH' ,  'SNUH_test') :

    df_val = pd.read_csv(path + "/val.csv")
    df_val= df_val.drop(columns=drop)

elif run == 'KMC' :

    drop = ['Others','sex']
    df_val = pd.read_csv(path +"/kmc_val.csv")
    df_val = df_val.drop(columns = drop)

else :
    print("input run error")


feature_columns = df_train.columns
label = 'new_total_aki'
print(feature_columns)


train_df = df_train
dev_df = df_val[feature_columns]
test_df = df_test[feature_columns]

print("train",train_df.columns)
print("val",dev_df.columns)
print("test",test_df.columns)

print('Number of training samples:', len(train_df))
print('Number of dev samples:', len(dev_df))
print('Number of test samples:', len(test_df))



# Main model

for name, metric in metrics.items():
    #make folder if doens't exist
    save_path = output_path+str(run)+'/'+str(name)
    os.makedirs(save_path, exist_ok=True)

    tm = localtime()
    print(">>>>starting at: ", strftime('%Y-%m-%d %I:%M:%S %p', tm))
    start = time()

    path_model_name = 'notext+normal+batch56+conv'+name


    print("[fitting",name, "start]")

#train MM

    if os.path.exists(path_model_name):
        predictor = MultiModalPredictor.load(path_model_name)
        print("file exists, load complete")
    else :
        print("--------no existing file found, new train start--------")
        predictor = MultiModalPredictor(label='new_total_aki', eval_metric=metric, path = path_model_name)  #alldata+metric
        predictor.fit(train_df, hyperparameters=hyp )
    print(predictor.fit_summary(verbosity=4, show_plot=True))

    print(name, "train complete, start evaluation")

#result
    performance = predictor.evaluate(test_df, metrics=['roc_auc','f1', 'specificity', 'recall', 'precision' ,'accuracy'])
    print(performance)

    eval_dict= {}
    label_col = 'new_total_aki'
    score_col = 'score'

    if task=='test' :
        
        df_groups = {
            'train': train_df,
            'test': test_df }
        
        for df_name, df_group in df_groups.items():
        
            labels_df = df_group[label_col].values
            pred_df =predictor.predict(df_group[input_cols])

            proba_df  = predictor.predict_proba(df_group[input_cols]).iloc[:, 1]
            list = 'transformer+notext' + '_' + df_name
            
            eval_dict[list] = pd.DataFrame({
                            label_col: labels_df,
                            score_col: proba_df,
                            'y_pred' : pred_df })

    elif task =='eval' :
#val data
        labels_val = dev_df[label_col].values
        pred_val =predictor.predict(dev_df[input_cols])

        proba_val  = predictor.predict_proba(dev_df[input_cols]).iloc[:, 1]

        list = 'transformer+notext' + '_val'

        eval_dict[list] = pd.DataFrame({
                            label_col: labels_val,
                            score_col: proba_val,
                            'y_pred' : pred_val })
#platt scaling
        list =  'transformer+notext' + '+platt'
        labels_test = test_df[label_col].values
        proba_test =predictor.predict_proba(test_df[input_cols]).iloc[:, 1]
        pred_test =predictor.predict(test_df[input_cols])

        platt = PlattCalibrator(log_odds=True)
        platt.fit(proba_val.values, labels_val)
        platt_probs = platt.predict(proba_test.values)

        eval_dict[list] = pd.DataFrame({
                                label_col: labels_test,
                                score_col: platt_probs,
                                'y_pred' : pred_test })



    print("start printing results")
    n_bins = 15
    
#proba    
    if os.path.isfile(save_path+'/auc.csv') :
        d = pd.read_csv(save_path+'/auc.csv')
    else :
        d = df_val['new_total_aki']

    for key, value in eval_dict.items() :
        if key.split("_")[-1] == 'val' :
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

