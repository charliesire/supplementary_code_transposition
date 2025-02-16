import os, sys
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import dataframe_image as dfi


#The function plot_mean_std plots the mean and standard deviations for each method, and compares it to the true values and the calibration measures. 

#The function compare_errors gathers the RMSRE and levels of prediction intervals of all method, and compute $p^{0.9}_{N,M}$.

#The function plot_errors plots the RMSRE and $p^{0.9}_{N,M}$ for each method.

variable_names = [r"$\ell_{F} - \ell_{0}$", r"$r_{F}$", r"$\epsilon_{max}$"] #names of the three outputs

def plot_mean_std(index_calib, results_measures, true_values, sigma, pre_path, no_error = False, unif_error = False, hierarchical_map = False, full_bayes = False, embed = False, savefig = False):

    list_values = [true_values, results_measures]
    list_sigma = [[0]*3, sigma]
    list_labels = ["True value", "Measurement"]

    incr = 0.09 #distance between the error bars
    if hierarchical_map: plot_hierarchical_plugin = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) #get mean and std for hierarchical plugin
    if full_bayes: plot_full_bayes = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/full_bayes.csv", index_col=0) # get mean and std for hierarchical full bayes
    if no_error: plot_no_error = pd.read_csv(pre_path + f"no_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) # get mean and std for no error
    if unif_error: plot_unif_error = pd.read_csv(pre_path + f"uniform_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) #get mean and std for uniform error
    elinewidth = 3 #error bar width
    markersize = 8 
    if embed:
        Ysimu_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/predictions.csv", index_col=0) #get mean for embedded discrepancy
        Ystd_embed = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/std_dev.csv", index_col=0) #get std for embedded discrepancy
    
    x = np.arange(len(results_measures))
    ticks = [f"$x_{{{k+1}}}$" for k in x]

    fig, axes = plt.subplots(3, 1, figsize=(30, 13))  # 3 subplots 
    
    for i, ax in enumerate(axes, start=1):
        sum_increments = 0
        if index_calib == i: #if the output is the one observed
            ax.errorbar(x, (list_values[1][f"Y{i}"]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=list_sigma[1][i-1]/list_values[0][f"Y{i}"], fmt='o', color='blue', label=list_labels[1],elinewidth=elinewidth, markersize = markersize) #plot measures and variance noise
        
        ax.scatter(x, list_values[0][f"Y{i}"]-list_values[0][f"Y{i}"], marker='x', color='blue', label=list_labels[0], s=120,linewidths=4) #plot true values
        if no_error: 
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_no_error.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_no_error.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='green', label='No error',elinewidth=elinewidth, markersize = markersize)  #plot no_error
        if unif_error: 
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_unif_error.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_unif_error.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='red', label='Uniform error',elinewidth=elinewidth,markersize = markersize) #plot uniform error
        if hierarchical_map:
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_hierarchical_plugin.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_hierarchical_plugin.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='purple', label="Hierarchical \n     MAP",elinewidth=elinewidth,markersize = markersize) #plot hierarchical map
        if full_bayes:
            sum_increments += 1
            ax.errorbar(x + sum_increments*incr, (plot_full_bayes.iloc[:10, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=plot_full_bayes.iloc[10:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='magenta', label="Hierarchical \n full Bayes",elinewidth=elinewidth,markersize = markersize) #plot full bayes
        if embed: 
            sum_increments += 1
            ax.errorbar(x + 5*incr, (Ysimu_embed.iloc[:, i-1]-list_values[0][f"Y{i}"])/list_values[0][f"Y{i}"], yerr=Ystd_embed.iloc[:, i-1]/list_values[0][f"Y{i}"], fmt='o', color='orange', label="Embedded \ndiscrepancy",elinewidth=elinewidth,markersize = markersize) #plot embedded discrepancy
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

        ax.set_title(f"Prediction of {variable_names[i-1]} from measures of {variable_names[index_calib-1]}", fontsize=42)
        
        if i == len(axes): #x ticks on the last subplots
            ax.set_xticks(x)
            ax.set_xticklabels(ticks, fontsize=30)
        else:
            ax.set_xticks([])  # Remove x-ticks for the first two subplots
        
        ax.tick_params(axis='y', labelsize=20)
    
    handles, labels = axes[index_calib-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', fontsize=30, bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    if savefig: 
        if(os.path.isdir(pre_path + "plots")==False):  
            os.mkdir(pre_path + "plots")
        plt.savefig(pre_path + f"plots/plot_pred_{index_calib}.jpg",bbox_inches='tight',format='jpg')    
    plt.show()
    
def get_score(df_mean_std, true_values):
    return -np.log(df_mean_std.iloc[10:,]**2)-(true_values.iloc[:,3:6].values-df_mean_std.iloc[:10,])**2/df_mean_std.iloc[10:,]**2



def get_table_score(index_calib, pre_path, true_values, no_error = True, unif_error = True, hierarchical_map = True, full_bayes = True, embed = True, savefig = False):
    names = []
    res = np.empty((3,0))
    elinewidth = 3 #error bar width
    markersize = 8 
    if no_error:
        plot_no_error = pd.read_csv(pre_path + f"no_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) # get mean and std for no error
        score_no_error = get_score(plot_no_error, true_values)
        res = np.round(np.concatenate([res, np.sum(score_no_error, axis = 0).values.reshape(-1,1)], axis = 1),2)
        names.append("No error")
    if unif_error:
        plot_unif_error = pd.read_csv(pre_path + f"uniform_error/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv", index_col=0) #get mean and std for uniform error
        score_unif_error = get_score(plot_unif_error, true_values)
        res = np.round(np.concatenate([res, np.sum(score_unif_error, axis = 0).values.reshape(-1,1)], axis = 1),2)
        names.append("Uniform error")
    if hierarchical_map:
        plot_hierarchical_plugin = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/plot_alpha_map_lamdba_bayesian.csv",index_col=0) #get mean and std for hierarchical plugin  
        score_hier_plugin = get_score(plot_hierarchical_plugin, true_values)
        res = np.round(np.concatenate([res, np.sum(score_hier_plugin, axis = 0).values.reshape(-1,1)], axis = 1),2)
        names.append("Hierarchical MAP")
    if full_bayes:
        plot_full_bayes = pd.read_csv(pre_path + f"hierarchical_model/calib_{index_calib}/full_bayes.csv", index_col=0) # get mean and std for hierarchical full bayes
        score_full_bayes = get_score(plot_full_bayes, true_values)
        res = np.round(np.concatenate([res, np.sum(score_full_bayes, axis = 0).values.reshape(-1,1)], axis = 1),2)
        names.append("Hierarchical full Bayes")
    if embed:
        mean_full_bayes = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/predictions.csv", index_col=0) # get mean and std for hierarchical full bayes
        std_full_bayes = pd.read_csv(pre_path + f"embedded_discrepancy/calib_{index_calib}/std_dev.csv", index_col=0) # get mean and std for hierarchical full bayes
        plot_full_bayes = pd.concat([mean_full_bayes, std_full_bayes], axis = 0)
        score_full_bayes = get_score(plot_full_bayes, true_values)
        res = np.round(np.concatenate([res, np.sum(score_full_bayes, axis = 0).values.reshape(-1,1)], axis = 1),2)
        names.append("Embedded discrepancy")

    res = pd.DataFrame(res, columns = names)

    if savefig:
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.axis('tight')
        ax.axis('off')
    
        table = ax.table(cellText=res.values, colLabels=res.columns, cellLoc='center', loc='center')
    
        plt.savefig(pre_path + f"plots/scores_{index_calib}.jpg",bbox_inches='tight',format='jpg')
        plt.close()

    return res
    
