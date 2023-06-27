import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from data_preprocessing import MorganECFPCounts
from utils import fingerprints_from_mol
from scripts.simulated_expert import logPEvaluationModel
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

feedback_model = logPEvaluationModel()

def get_scaffold_memory_results(output_dir, reinvent_rounds, acquisition = "random", n_seeds = 1):
    #return dict with keys "seed" and values lists of reinvent_rounds scaffold memory dataframes
    output_dir = f"/home/klgx638/Generations/HITL_qsar_experiments/demo_logpSmall_{acquisition}_noise0.0"
    scaffs = dict()
    for seed in range(1, n_seeds + 1):
        if acquisition == "None":
            path = f"{output_dir}_seed{seed}/results/scaffold_memory.csv"
            scaff_0 = pd.read_csv(path)
        else:
            path = f"{output_dir}_seed{seed}/iteration_0/scaffold_memory.csv"
            scaff_0 = pd.read_csv(path)
        scaffs[seed] = [scaff_0]
        for i in range(1, reinvent_rounds):
            path_i = f"{output_dir}_seed{seed}/iteration{i}_{acquisition}/results/scaffold_memory.csv"
            scaff_i = pd.read_csv(path_i)
            scaffs[seed].append(scaff_i)
    return scaffs

def get_oracle_scores(df, score_component_name, top_mols):
    sorted_compounds = df.sort_values(by=[score_component_name], ascending=False)
    if top_mols:
        sorted_compounds = sorted_compounds.iloc[:top_mols]
    oracle_scores = [feedback_model.oracle_score(s) for s in sorted_compounds.SMILES] #replace with oracle from TDC benchmark
    return oracle_scores

def get_raw_scores(df, score_component_name, top_mols, narrow_space = True):
    sorted_compounds = df.sort_values(by=[score_component_name], ascending=False)
    if top_mols:
        sorted_compounds = sorted_compounds.iloc[:top_mols]
    raw_scores = sorted_compounds[f"raw_{score_component_name}"].values.tolist()
    return raw_scores

def get_average_score_by_iter(scaff_dict, score_component_name, reinvent_rounds, top_mols = None):
    avg_score_by_iter = dict()
    for iteration in range(reinvent_rounds):
        avg_score_by_seed = []
        for seed in scaff_dict.keys():
            scaff = scaff_dict[seed][iteration]
            oracle_scores = get_oracle_scores(scaff, score_component_name, top_mols)
            utilities = []
            for score in oracle_scores:
                try:
                    utility = feedback_model.utility(score, low = 2, high = 4)
                except:
                    print("Overflow error")
                    utility = feedback_model.utility(round(score), low = 2, high = 4)
                utilities.append(utility)
            avg_score_by_seed.append(np.mean(utilities))
        avg_score_by_iter[iteration] = avg_score_by_seed
            
    return avg_score_by_iter

def get_avg_accuracy_by_iter(scaff_dict, metric, score_component_name, reinvent_rounds, top_mols = None):
    avg_score_by_iter = dict()
    for iteration in range(reinvent_rounds):
        avg_score_by_seed = []
        accs = []
        for seed in scaff_dict.keys():
            scaff = scaff_dict[seed][iteration]
            oracle_scores = get_oracle_scores(scaff, score_component_name, top_mols)
            raw_scores = get_raw_scores(scaff, score_component_name, top_mols)
            if metric == "mean squared error":
                accs.append(mean_absolute_error(oracle_scores, raw_scores))
            if metric == "pearson correlation":
                accs.append(pearsonr(oracle_scores, raw_scores)[0])
        avg_score_by_iter[iteration] = accs
            
    return avg_score_by_iter
    

def get_all_acquisitions(output_dir, list_acquisitions, reinvent_rounds, n_seeds):
    res_dict = dict()
    for acq in list_acquisitions:
        res_dict[acq] = get_scaffold_memory_results(output_dir, reinvent_rounds, acq, n_seeds)
    return res_dict
    
def get_means_of_repetitions(avg_score_by_iter, reinvent_rounds):
    means_by_iter = [np.mean(avg_score_by_iter[i]) for i in range(reinvent_rounds)]
    sd_by_iter = [np.std(avg_score_by_iter[i]) for i in range(reinvent_rounds)]
    
    return means_by_iter, sd_by_iter

def plot_oracle_score(
    baseline_output_dir, 
    compared_output_dir,  
    acquisition_method = ["random"],
    score_component_name = "bioactivity", 
    n_seeds = 1,
    reinvent_rounds = 10,
    top_mols = None, 
    include_predicted_score = False,
    include_correlation_score = False,
    legend_drawn_flag = True,
    draw_molecules = False,
    save_to_path = "/home/klgx638/Generations/HITL_qsar_experiments/figures"
    ):
    
    scaffs_baseline = get_scaffold_memory_results(baseline_output_dir, reinvent_rounds, acquisition = "None", n_seeds = n_seeds)
    avg_score_by_iter_baseline = get_average_score_by_iter(scaffs_baseline, score_component_name, reinvent_rounds, top_mols)
    means_by_iter_baseline, sd_by_iter_baseline = get_means_of_repetitions(avg_score_by_iter_baseline, reinvent_rounds)
    
    if len(acquisition_method) > 1:
        avg_score_by_iter_compared = []
        scaffs_compared = get_all_acquisitions(compared_output_dir, acquisition_method, reinvent_rounds, n_seeds)
        for acq in acquisition_method:
            for seed in range(1, n_seeds + 1):
                assert scaffs_baseline[seed][0].total_score.values.mean() == scaffs_compared[acq][seed][0].total_score.values.mean()
            avg_score_by_iter_compared.append(get_average_score_by_iter(scaffs_compared[acq], score_component_name, reinvent_rounds, top_mols))
            
    else:
        acq = acquisition_method[0]
        scaffs_compared = get_scaffold_memory_results(compared_output_dir, reinvent_rounds, acq, n_seeds = n_seeds)
        for seed in range(1, n_seeds + 1):
            assert scaffs_baseline[seed][0].total_score.values.mean() == scaffs_compared[seed][0].total_score.values.mean()
        avg_score_by_iter_baseline = get_average_score_by_iter(scaffs_baseline, score_component_name, reinvent_rounds, top_mols)
    
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.tight_layout()

    #plot oracle score evolution
    
    if n_seeds > 1:
    
        ax[0].plot(np.arange(reinvent_rounds), means_by_iter_baseline)
    
        for i in range(len(avg_score_by_iter_compared)):
            means_by_iter_compared, sd_by_iter_compared = get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)
            ax[0].plot(np.arange(reinvent_rounds), means_by_iter_compared)
    
            
        
        ax[0].fill_between(
                np.arange(reinvent_rounds), 
                np.array(means_by_iter_baseline) - np.array(sd_by_iter_baseline), 
                np.array(means_by_iter_baseline) + np.array(sd_by_iter_baseline), 
                alpha=.15
                )
        
        for i in range(len(avg_score_by_iter_compared)):
            ax[0].fill_between(
                            np.arange(reinvent_rounds), 
                            np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[0]) - np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[1]), 
                            np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[0]) + np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[1]), 
                            alpha=.15
                            )
    else:
        ax[0].plot(np.arange(reinvent_rounds), means_by_iter_baseline)
        ax[0].plot(np.arange(reinvent_rounds), means_by_iter_compared)
        ax[0].fill_between(
                np.arange(reinvent_rounds), 
                np.array(means_by_iter_baseline) - np.array(sd_by_iter_baseline), 
                np.array(means_by_iter_baseline) + np.array(sd_by_iter_baseline), 
                alpha=.15
                )
        ax[0].fill_between(
                    np.arange(reinvent_rounds), 
                    np.array(means_by_iter_compared) - np.array(sd_by_iter_compared), 
                    np.array(means_by_iter_compared) + np.array(sd_by_iter_compared), 
                    alpha=.15
                    )
        
    
    if include_predicted_score:
        ax[0].plot(
            np.arange(reinvent_rounds), 
            [feedback_model.utility(np.mean(scaffs_baseline[seed][i][f"raw_{score_component_name}"]), low = 2, high = 4) for i in range(reinvent_rounds)], 
            linestyle = "--",
            color = "b"
        )
        ax[0].plot(
            np.arange(reinvent_rounds), 
            [feedback_model.utility(np.mean(scaffs_compared[seed][i][f"raw_{score_component_name}"]), low = 2, high = 4) for i in range(reinvent_rounds)], 
            linestyle = "--",
            color = "orange"
        )
    
    if include_correlation_score:
        ax[0].set_xticks(np.arange(-1, reinvent_rounds))
    else:
        ax[0].set_xticks(np.arange(reinvent_rounds))
        
    if include_correlation_score:
        ax[0].plot(np.arange(-1, reinvent_rounds), [pearsonr(y_test.flatten(), y_pred_test0)[0]] + score_baseline[:1000])
        ax[0].plot(np.arange(-1, reinvent_rounds), [pearsonr(y_test.flatten(), y_pred_test)[0]] + score_compared[:1000])

    ax[0].set_ylim(ymin = 0)
    ax[0].set_ylim(ymax = 1.01)
    
    if len(acquisition_method) > 1:
        ax[0].legend(["no human"] + acquisition_method, loc=0, frameon=legend_drawn_flag)
    else:
        ax[0].legend(["no human", f"{acquisition_method}"], loc=0, frameon=legend_drawn_flag)
    if include_predicted_score:
        ax[0].legend(["no human (oracle score)", f"{acquisition_method} (oracle score)", "no human (predicted score)", f"{acquisition_method} (predicted score)"], 
                     loc=0, frameon=legend_drawn_flag)
    if include_correlation_score:
        ax[0].legend(["no human (oracle score)", f"{acquisition_method} (oracle score)", "no human (pred vs. oracle correlation)", f"{acquisition_method} (pred vs. oracle correlation)"], 
                     loc=0, frameon=legend_drawn_flag)
    if include_predicted_score and include_correlation_score:
        ax[0].legend(["no human (oracle score)", f"{acquisition_method} (oracle score)", "no human (predicted score)", f"{acquisition_method} (predicted score)", "no human (pred vs. oracle correlation)", f"{acquisition_method} (pred vs. oracle correlation)"], 
                     loc=0, frameon=legend_drawn_flag)
    ax[0].axhline(y=0.5, color = "gray", linestyle = "--")
    ax[0].set_ylabel("Score")
    ax[0].set_xlabel("Human interaction round (R)")
    ax[0].set_title("Oracle score evolution on batches of generated molecules at each round")

    #plot oracle scores distribution for last 100 rounds
    #for seed in scaffs_compared.keys():
    baseline_oracle_scores, baseline_means = [], []
    compared_oracle_scores, compared_means = [], []
    for seed in range(1, n_seeds + 1):
        baseline_oracle_scores.append(get_oracle_scores(scaffs_baseline[seed][-1], score_component_name, top_mols))
        compared_oracle_scores.append(get_oracle_scores(scaffs_compared[acq][seed][-1], score_component_name, top_mols))
        baseline_means.append(np.mean(baseline_oracle_scores))
        compared_means.append(np.mean(compared_oracle_scores))
        ax[1].hist(baseline_oracle_scores[seed-1], alpha=.3, color = "blue")
        ax[1].hist(compared_oracle_scores[seed-1], alpha=.3, color = "orange")
    ax[1].axvline(x=2, color = "g", linestyle = "--")
    ax[1].axvline(x=4, color = "g", linestyle = "--")
    ax[1].axvline(x=np.mean(baseline_means), color = "blue")
    ax[1].axvline(x=np.mean(compared_means), color = "orange")
    ax[1].legend(["no human", f"{acquisition_method}"], loc=0, frameon=legend_drawn_flag)
    ax[1].set_xlim((0,8))
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Oracle score")
    ax[1].set_title(f"Oracle scores distrbution after last optimization round R = {reinvent_rounds}") 

    fig.savefig(os.path.join(f"{save_to_path}", f"results_{acquisition_method}.png"))
    
def plot_accuracy(
    baseline_output_dir, 
    compared_output_dir,  
    acquisition_method = ["random"],
    metrics = "mean squared error", # or pearson correlation score
    component_name = "bioactivity", 
    n_seeds = 1,
    reinvent_rounds = 10,
    top_mols = None,
    legend_drawn_flag = True,
    save_to_path = "/home/klgx638/Generations/HITL_qsar_experiments/figures"
    ):
    
    scaffs_baseline = get_scaffold_memory_results(baseline_output_dir, reinvent_rounds, acquisition = "None", n_seeds = n_seeds)
    avg_score_by_iter_baseline = get_avg_accuracy_by_iter(scaffs_baseline, metric, component_name, reinvent_rounds, top_mols)
    means_by_iter_baseline, sd_by_iter_baseline = get_means_of_repetitions(avg_score_by_iter_baseline, reinvent_rounds)
    
    if len(acquisition_method) > 1:
        avg_score_by_iter_compared = []
        scaffs_compared = get_all_acquisitions(compared_output_dir, acquisition_method, reinvent_rounds, n_seeds)
        for acq in acquisition_method:
            for seed in range(1, n_seeds + 1):
                assert scaffs_baseline[seed][0].total_score.values.mean() == scaffs_compared[acq][seed][0].total_score.values.mean()
            avg_score_by_iter_compared.append(get_avg_accuracy_by_iter(scaffs_compared[acq], metric, component_name, reinvent_rounds, top_mols))
    
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.tight_layout()
    
    ax[0].plot(np.arange(reinvent_rounds), means_by_iter_baseline)
    
    for i in range(len(avg_score_by_iter_compared)):
        means_by_iter_compared, sd_by_iter_compared = get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)
        ax[0].plot(np.arange(reinvent_rounds), means_by_iter_compared)
    
            
        
    ax[0].fill_between(
        np.arange(reinvent_rounds), 
        np.array(means_by_iter_baseline) - np.array(sd_by_iter_baseline), 
        np.array(means_by_iter_baseline) + np.array(sd_by_iter_baseline), 
        alpha=.15
    )
        
    for i in range(len(avg_score_by_iter_compared)):
        ax[0].fill_between(
                np.arange(reinvent_rounds), 
                np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[0]) - np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[1]), 
                np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[0]) + np.array(get_means_of_repetitions(avg_score_by_iter_compared[i], reinvent_rounds)[1]), 
                alpha=.15
            )
            
    ax[0].legend(["no human"] + acquisition_method, loc=0, frameon=legend_drawn_flag)
    ax[0].set_ylabel(f"{metric}")
    ax[0].set_xlabel("Human interaction round (R)")
    ax[0].set_title("Predictive accuracy on generated molecules per round")
