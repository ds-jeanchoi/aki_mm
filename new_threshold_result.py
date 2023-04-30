import stat_util
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix,  auc, precision_recall_curve, brier_score_loss

metrics = ["auroc", "f1", "speci", "sensit", "AUPRC"]
round_digits  = 3


df = pd.read_csv("./final_output_0429/SNUH/f1/auc.csv") #SNUH -> calculate optimal threshold using best f1-score


res = {}




def new_metric(task):
    # ./m1/SNUH/
    path = './final_output_0429/'+str(task)+"/"
    
    for m in range(len(metrics)) :
        print(metrics[m])
        
        #fix threshold with SNUH
        
        cutoff  = list(np.arange(0.0,1.0,0.01))
        dfs = pd.read_csv('./final_output_0429/SNUH/'+metrics[m]+"/auc.csv") 
        y_test = dfs['new_total_aki']
        for col in dfs.columns.drop('new_total_aki') :
            l = {}
            for c in cutoff :   
                roc_predictions = [1 if i >= c else 0 for i in dfs[col]]
                l.update({round(c,2): round(f1_score(y_test, roc_predictions),4)})
            optimal_threshold = max(l, key=l.get)
            res[col] = optimal_threshold 
        print(res)    
            
        #import data use threshold from SNUH
  
        if task == "SNUH" :
            df = pd.read_csv(path+metrics[m]+"/auc.csv") #SNUH -> calculate optimal threshold 
            #y_test = df['new_total_aki']
            
#             res = {}
            
#             for col in df.columns.drop('new_total_aki') :
#                 l = {}
#                 for c in cutoff :   
#                     roc_predictions = [1 if i >= c else 0 for i in df[col]]
#                     l.update({round(c,2): round(f1_score(y_test, roc_predictions),4)})
#                 optimal_threshold = max(l, key=l.get)
#                 res[col] = optimal_threshold 

        elif task == "KMC" :
            df = pd.read_csv(path+metrics[m]+"/auc.csv")  #KMC
          #  col_list = df.columns[1:]
          #  res = {}
            
        new_metrics = []



        for col, cutoff in res.items():
    
            y_true = df['new_total_aki']
            y_pred =  [1 if i >= cutoff else 0 for i in df[col]]
            y_prob = df[col]

            auroc = round(roc_auc_score(y_true, y_prob), round_digits)
            brier_score = round(brier_score_loss(y_true, y_prob), round_digits)


            precisions, recalls, threshold = precision_recall_curve(y_true, y_prob)


            score_auroc, ci_lower_auroc, ci_upper_auroc, scores_auroc = stat_util.score_ci(y_true, y_prob,
                                                                           score_fun=roc_auc_score,
                                                                           n_bootstraps=1000,
                                                                           seed=42)
            score_prauc, ci_lower_prauc, ci_upper_prauc, scores_prauc = stat_util.score_ci(recalls, precisions, score_fun=auc)

            auc_ci = str(round(score_auroc,3))+ " (" + str(round(ci_lower_auroc,3)) + "-" + str(round(ci_upper_auroc,3)) + ")"
            auprc_ci = str(round(score_prauc, 3)) + " (" + str(round(ci_lower_prauc, 3)) + "-" + str(round(ci_upper_prauc, 3)) + ")"


            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            sensitivity =  recall
            specificity = tn / (tn+fp)
            #auprc = auc(recall, precision)

            
#             if  col.split('_')[0] == 'RF' :
#                 new_name = 'Random Forest'
#             elif col.split('_')[0] == 'LR' :
#                 new_name = 'Logistic Regression'
#             elif col.split('_')[0] == 'xgb':
#                 new_name = 'XGBoost'
#             elif col.split('_')[0] == 'svc':
#                 new_name = 'SVM'
#             elif col.split('_')[0] == 'transformer'  :
#                 new_name = 'Transformer'
#             elif col.split('_')[0] == 'transformer+notext' :
#                 new_name = 'Transformer+notext'

#             if col.split('_')[-1] in ( 'train','test') :
#                 new_name= new_name+" "+ col.split('_')[1]
            
            
            new_metrics.append({ 'name' : col, #new_name,
                'thresholds' : cutoff,               
                'auroc': round(auroc,3),
                'auroc+ci' : auc_ci,
                'auprc+ci' : auprc_ci,
                'precision': round(precision,3),
                'sensitivity': round(sensitivity,3),
                'specificity' : round(specificity,3),
                'f1': round(f1,3),
                'brier': round(brier_score,3)
            })
        final_metrics =  pd.DataFrame.from_dict(new_metrics)
        final_metrics.to_csv(path+metrics[m]+"/new_threshold_result.csv")
        

#new_metric("SNUH")
new_metric("KMC")
print("finished")