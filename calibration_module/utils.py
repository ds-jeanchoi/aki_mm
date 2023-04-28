import os
import math
import numpy as np
import pandas as pd
import stat_util
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from typing import Dict, List, Tuple, Optional
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    f1_score,
    recall_score,
    roc_curve,
    precision_score,
    precision_recall_curve,
    roc_auc_score,
    confusion_matrix)
from calibration import CalibrationBelt



__all__ = [
    'compute_calibration_error',
    'create_binned_data',
    'get_bin_boundaries',
    'compute_binary_score',
    'compute_calibration_summary'
]





def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int=15,
    round_digits: int=4) -> float:
    """
    Computes the calibration error for binary classification via binning
    data points into the specified number of bins. Samples with similar
    ``y_prob`` will be grouped into the same bin. The bin boundary is
    determined by having similar number of samples within each bin.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    n_bins : int, default 15
        A bigger bin number requires more data. In general,
        the larger the bin size, the closer the calibration error
        will be to the true calibration error.

    round_digits : int, default 4
        Round the calibration error metric.

    Returns
    -------
    calibration_error : float
        RMSE between the average positive label and predicted probability
        within each bin.
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)

    binned_y_true, binned_y_prob = create_binned_data(y_true, y_prob, n_bins)

    # looping shouldn't be a source of bottleneck as n_bins should be a small number.
    bin_errors = 0.0
    for bin_y_true, bin_y_prob in zip(binned_y_true, binned_y_prob):
        avg_y_true = np.mean(bin_y_true)
        avg_y_score = np.mean(bin_y_prob)
        bin_error = (avg_y_score - avg_y_true) ** 2
        bin_errors += bin_error * len(bin_y_true)

    calibration_error = math.sqrt(bin_errors / len(y_true))
    return round(calibration_error, round_digits)


def create_binned_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Bin ``y_true`` and ``y_prob`` by distribution of the data.
    i.e. each bin will contain approximately an equal number of
    data points. Bins are sorted based on ascending order of ``y_prob``.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    n_bins : int, default 15
        A bigger bin number requires more data.

    Returns
    -------
    binned_y_true/binned_y_prob : 1d ndarray
        Each element in the list stores the data for that bin.
    """
    sorted_indices = np.argsort(y_prob)
    sorted_y_true = y_true[sorted_indices]
    sorted_y_prob = y_prob[sorted_indices]
    binned_y_true = np.array_split(sorted_y_true, n_bins)
    binned_y_prob = np.array_split(sorted_y_prob, n_bins)
    return binned_y_true, binned_y_prob


def get_bin_boundaries(binned_y_prob: List[np.ndarray]) -> np.ndarray:
    """
    Given ``binned_y_prob`` from ``create_binned_data`` get the
    boundaries for each bin.

    Parameters
    ----------
    binned_y_prob : list
        Each element in the list stores the data for that bin.

    Returns
    -------
    bins : 1d ndarray
        Boundaries for each bin.
    """
    bins = []
    for i in range(len(binned_y_prob) - 1):
        last_prob = binned_y_prob[i][-1]
        next_first_prob = binned_y_prob[i + 1][0]
        bins.append((last_prob + next_first_prob) / 2.0)

    bins.append(1.0)
    return np.array(bins)


def compute_binary_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    round_digits: int=4) -> Dict[str, float]:
    """
    Compute various evaluation metrics for binary classification.
    Including auc, precision, recall, f1, log loss, brier score. The
    threshold for precision and recall numbers are based on the one
    that gives the best f1 score.

    Parameters
    ----------
    y_true : 1d ndarray
        Binary true targets.

    y_prob : 1d ndarray
        Raw probability/score of the positive class.

    round_digits : int, default 4
        Round the evaluation metric.

    Returns
    -------
    metrics_dict : dict
        Metrics are stored in key value pair. ::

        {
            'auc': 0.82,
            'precision': 0.56,
            'recall': 0.61,
            'f1': 0.59,
            'log_loss': 0.42,
            'brier': 0.12
        }
    """
    auc = round(metrics.roc_auc_score(y_true, y_prob), round_digits)
#    log_loss = round(metrics.log_loss(y_true, y_prob), round_digits)
    brier_score = round(metrics.brier_score_loss(y_true, y_prob), round_digits)

    
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall)
    

##############added #################
    #new prediction using optimal threshold

    cutoff  = list(np.arange(0.0,1.0,0.01))  #find optimal threshold between 0~1 by 0.01

    l = {}
    for c in cutoff :    
        roc_predictions = [1 if i >= c else 0 for i in y_prob]
        l.update({round(c,2): round(f1_score(y_true, roc_predictions),4)})

    optimal_threshold = max(l, key=l.get)
    y_pred = [1 if i >= optimal_threshold  else 0 for i in y_prob]

####################################

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn+fp)
    
    auprc = metrics.auc(recall, precision)
    mask = ~np.isnan(f1)
    f1 = f1[mask]
    precision = precision[mask]
    recall = recall[mask]
#   specificity = specificity[mask]
   
    #print(mask)
    #massk =  pd.DataFrame(mask)



    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = metrics.confusion_matrix(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sensitivity =  recall
    
    
    print("f1", f1)
    print("recall", recall)
    print("precision", precision)
    
    
    
    #specificity =  cm[0,0]/(cm[0,0]+cm[0,1])
    return {
        'auroc': round(auc,3),
        'auprc' : round(auprc,3),
        'precision': round(precision,3),
    #    'recall': round(recall,3),
        'threshold' : round(optimal_threshold,3) ,
        'sensitivity': round(sensitivity,3),
        'specificity' : round(specificity,3),
        'f1': round(f1,3),
        'brier': round(brier_score,3)
    }



def compute_calibration_summary(
    eval_dict: Dict[str, pd.DataFrame],
    label_col: str='label',
    score_col: str='score',
    n_bins: int=15,
    strategy: str='quantile',
    round_digits: int=4,
    show: bool=True,
    save_plot_path: Optional[str]=None) -> pd.DataFrame:
    """
    Plots the calibration curve and computes the summary statistics for the model.

    Parameters
    ----------
    eval_dict : dict
        We can evaluate multiple calibration model's performance in one go. The key
        is the model name used to distinguish different calibration model, the value
        is the dataframe that stores the binary true targets and the predicted score
        for the positive class.

    label_col : str
        Column name for the dataframe in ``eval_dict`` that stores the binary true targets.

    score_col : str
        Column name for the dataframe in ``eval_dict`` that stores the predicted score.

    n_bins : int, default 15
        Number of bins to discretize the calibration curve plot and calibration error statistics.
        A bigger number requires more data, but will be closer to the true calibration error.

    strategy : {'uniform', 'quantile'}, default 'quantile'
        Strategy used to define the boundary of the bins.

        - uniform: The bins have identical widths.
        - quantile: The bins have the same number of samples and depend on the predicted score.

    round_digits : default 4
        Round the evaluation metric.

    show : bool, default True
        Whether to show the plots on the console or jupyter notebook.

    save_plot_path : str, default None
        Path where we'll store the calibration plot. None means it will not save the plot.

    Returns
    -------
    df_metrics : pd.DataFrame
        Corresponding metrics for all the input dataframe.
    """

    fig1, ax1 = plt.subplots(1)
    fig2, ax2 = plt.subplots(1)
    fig3, ax3 = plt.subplots(1)
    fig6, ax6 = plt.subplots(1)
    
    estimator_metrics = []
    for name, df_eval in eval_dict.items():
        prob_true, prob_pred = calibration_curve(
            df_eval[label_col],
            df_eval[score_col],
            n_bins=n_bins,
            strategy=strategy)
       
        
        if  name.split('_')[0] == 'RF' :
            new_name = 'Random Forest'
        elif name.split('_')[0] == 'LR' :
            new_name = 'Logistic Regression'
        elif name.split('_')[0] == 'xgb':
            new_name = 'XGBoost'
        elif name.split('_')[0] == 'svc':
            new_name = 'SVM'
        elif name.split('_')[0] == 'transformer'  :
            new_name = 'Transformer'
        elif name.split('_')[0] == 'transformer+notext' :
            new_name = 'Transformer+notext'
        
        if name.split('_')[-1] in ( 'train','test') :
            new_name= new_name+" "+ name.split('_')[1] 

        ### add auprc, auroc curve +CI
        precisions, recalls, thresholds = precision_recall_curve(df_eval[label_col],
                                                                 df_eval[score_col])
        fpr, tpr, thresholds = roc_curve(df_eval[label_col],
                                         df_eval[score_col])


        #ci 추가
        score_auroc, ci_lower_auroc, ci_upper_auroc, scores_auroc = stat_util.score_ci(df_eval[label_col], df_eval[score_col],
                                                               score_fun=roc_auc_score,
                                                               n_bootstraps=1000,
                                                               seed=42)
        score_prauc, ci_lower_prauc, ci_upper_prauc, scores_prauc = stat_util.score_ci(recalls, precisions, score_fun=auc)

        #for non-platt: plot ax1, for platt data: plot ax2
        
        
        if  name.split('+')[-1] not in ( 'platt', 'isotonic')  :
        
            calibration_error = compute_calibration_error(
                df_eval[label_col], df_eval[score_col], n_bins, round_digits)


            auc_ci = str(round(score_auroc,3))+ " (" + str(round(ci_lower_auroc,3)) + "-" + str(round(ci_upper_auroc,3)) + ")"
            auprc_ci = str(round(score_prauc, 3)) + " (" + str(round(ci_lower_prauc, 3)) + "-" + str(round(ci_upper_prauc, 3)) + ")"

            metrics_dict = compute_binary_score(df_eval[label_col], df_eval[score_col], df_eval['y_pred'], round_digits)
            metrics_dict['auroc+ci'] =  auc_ci
            metrics_dict['auprc+ci'] = auprc_ci
            metrics_dict['calibration_error'] = calibration_error
            metrics_dict['name'] = new_name
    #        metrics_dict['threshold'] = round(thresholds[np.argmax(tpr - fpr)],3)
    # add threshold
            estimator_metrics.append(metrics_dict)

            ax1.plot(prob_pred, prob_true, 's-', label=new_name, linewidth=1.0)
            ax2.plot(recalls, precisions, 's-', label=" %s (AUPRC = %.3f [%.3f - %.3f])" %(new_name, score_prauc, ci_lower_prauc, ci_upper_prauc), linewidth=1.0)    #auprc
            ax3.plot(fpr, tpr, 's-', label = " %s (AUROC = %.3f [%.3f - %.3f])" %(new_name, score_auroc, ci_lower_auroc, ci_upper_auroc), linewidth=1.0)    #auroc

            #calibration belt
            belt = CalibrationBelt(df_eval[label_col].values,df_eval[score_col].values)
            fig4, ax4 = belt.plot(confidences=[.8, .95], label= new_name)
            ax4.set_title('Calibration Curve,'+str(new_name), fontsize=20)
            fig4.savefig(save_plot_path + '/calib_belt_'+ str(new_name) +'.tif', dpi=300, bbox_inches='tight')
        # platt
        elif name.split('+')[-1]== "platt" :
            # platt-scaled calibration belt
            belt = CalibrationBelt(df_eval[label_col].values, df_eval[score_col].values)
            fig5, ax5 = belt.plot(confidences=[.8, .95], label=new_name)
            ax5.set_title('platt-scaled calibration,' + str(new_name), fontsize=20)
            fig5.savefig(save_plot_path + '/platt_belt_' + str(new_name) + '.tif', dpi=300, bbox_inches='tight')
            plt.close(fig5)

        else : #isotonic
            ax6.plot(prob_pred, prob_true, 's-', label=new_name, linewidth=1.0)
            ax6.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
            ax6.set_xlabel('Prediction')
            ax6.set_ylabel('Fraction of positives')
            ax6.legend(loc='lower right', ncol=1)
            ax6.set_title('isotonic calibration,' + str(new_name), fontsize=20)
            fig6.savefig(save_plot_path + '/isotonic_reg_' + str(new_name) + '.tif', dpi=300, bbox_inches='tight')
            plt.close(fig6)





    ax1.plot([0, 1], [0, 1], 'k:', label='perfect')

    ax1.set_xlabel('Fraction of positives (Predicted)')
    ax1.set_ylabel('Fraction of positives (Actual)')
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc='upper left', ncol=1)
    ax1.set_title('Calibration Plots (Reliability Curve)')
    #plt.show()

    ax2.set_xlabel('Recalls')
    ax2.set_ylabel('Precisions')
    ax2.set_ylim([-0.05, 1.05])
    ax2.legend(loc='upper right', ncol=1, fontsize=10)
    ax2.set_title('AUPRC')
    #plt.show()

    ax3.plot([0, 1], [0, 1], 'k:', label='reference')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_ylim([-0.05, 1.05])
    ax3.legend(loc='lower right', ncol=1, fontsize=10)
    ax3.set_title('AUROC')
    plt.tight_layout()
    #plt.show()

    #if show:
    #    plt.show()

    if save_plot_path is not None:
        save_dir = os.path.dirname(save_plot_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        
        
#save fig
    if new_name.split(' ')[0] not in ('Transformer', 'Transformer+notext')   :
        fig1.savefig(save_plot_path+'/ML_calibration_plot.tif', dpi=300, bbox_inches='tight')
        fig2.savefig(save_plot_path + '/ML_AUPRC_plot.tif', dpi=300, bbox_inches='tight')
        fig3.savefig(save_plot_path + '/ML_AUROC_plot.tif', dpi=300, bbox_inches='tight')

    else :
        fig1.savefig(save_plot_path+ "/"  +str(new_name)  +'_calibration_plot.tif', dpi=300, bbox_inches='tight')
        fig2.savefig(save_plot_path + "/"  +str(new_name) +  '_AUPRC_plot.tif', dpi=300, bbox_inches='tight')
        fig3.savefig(save_plot_path + "/"  +str(new_name) + '_AUROC_plot.tif', dpi=300, bbox_inches='tight')



    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)


    df_metrics = pd.DataFrame(estimator_metrics)
    #df_metrics.to_csv(save_plot_path+'/metrics.csv', mode='a')
    

    hdr = False  if os.path.isfile(save_plot_path+'/metrics.csv') else True
    df_metrics.to_csv(save_plot_path+'/metrics.csv', mode='a', header=hdr)

    print(df_metrics)
    return df_metrics
