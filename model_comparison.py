import os
import pandas as pd
import delong as dl
import numpy as np
import stat_util as st
from sklearn.metrics import average_precision_score

# final output folder from model.py
folder =  "final_output0224/SNUH/" 

metrics = ['f1', 'auroc', 'sensit','speci']

#save path for model comparison
save_path = "./final_output0224/model_comparison"
os.makedirs(save_path, exist_ok=True)
    
# AUROC delong test

for m in metrics :
    print(m)
    input_path =  str(folder) + "/" +str(m)
    d = pd.read_csv(input_path+'/pred.csv')
    cols = d.columns
    model_1 = []
    model_2 = []
    p_value = []
    for i in range(1,7) :
        for n in range(1,7) :
            if i == n :
                pass
            else:
                p = dl.delong_roc_test(d[cols[0]].values, d[cols[i]].values, d[cols[n]].values)[0][0]
                model_1.append(cols[i].split("_")[0])
                model_2.append(cols[n].split("_")[0])
                p_value.append(round(p,3))
    model_1 = pd.DataFrame(model_1)
    model_2 = pd.DataFrame(model_2)
    p_value = pd.DataFrame(p_value)
    p_list = pd.concat([model_1, model_2, p_value], axis=1)
    p_list.columns = ['model_1', 'model_2','p_value']
    df_pivoted = p_list.pivot(index='model_1', columns='model_2', values='p_value')
    df_pivoted.to_csv(save_path+"/auroc_delong_"+str(m)+'.csv')






#AUPRC value & CI & p_value

confidence_level= 0.95

for m in metrics :
    print(m)
    input_path =  str(folder) + "/" +str(m)
    d = pd.read_csv(input_path+'/auc.csv')
    cols = d.columns
    model_1 = []
    model_2 = []
    p_value = []
    for i in range(1,7) :
        for n in range(1,7) :
            if i == n :
                pass
            else:
                
                
                p, z = st.pvalue(d[cols[0]].values, d[cols[i]].values, d[cols[n]].values,
                        score_fun=average_precision_score,
                        seed=42)
                model_1.append(cols[i].split("_")[0])
                model_2.append(cols[n].split("_")[0])
                
                scores = z
                sorted_scores = np.array(sorted(scores))
                alpha = (1.0 - confidence_level) / 2.0
                ci_lower = round(sorted_scores[int(round(alpha * len(sorted_scores)))],3)
                ci_upper = round(sorted_scores[int(round((1.0 - alpha) * len(sorted_scores)))],3)
                
                P_ = (lambda x: "f" if x < 0.05 else "")
                value = str(round(sum(z)/len(z),3)) +" ("+ str(ci_lower) + " - " +str(ci_upper)  +") " + P_(p)
                p_value.append(value)
                
    model_1 = pd.DataFrame(model_1)
    model_2 = pd.DataFrame(model_2)
    p_value = pd.DataFrame(p_value)
    p_list = pd.concat([model_1, model_2, p_value], axis=1)
    p_list.columns = ['model_1', 'model_2','p_value']
    df_pivoted = p_list.pivot(index='model_1', columns='model_2', values='p_value')
    df_pivoted.to_csv(save_path+"/auprc_bs_"+str(m)+'.csv')
print("----------model comparison complete----------")
