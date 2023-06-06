import sys
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from data_preprocessing import MorganECFPCounts
from utils import fingerprints_from_mol
from scripts.simulated_expert_delta import logPEvaluationModel
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

feedback_model = logPEvaluationModel()

def get_scaffold_memory_results(output_dir, reinvent_rounds, acquisition = "random"):
    #return dict with keys "seed" and values lists of reinvent_rounds scaffold memory dataframes
    seed = int(output_dir.split("seed")[1])
    demo_dir = output_dir.split("demo_")[1]
    acquisition = demo_dir.split("_")[1]
    reinvent_rounds = 10
    if acquisition == "None":
        path = f"{output_dir}/results/scaffold_memory.csv"
        scaff_0 = pd.read_csv(path)
    else:
        path = f"{output_dir}/iteration_0/scaffold_memory.csv"
        scaff_0 = pd.read_csv(path)
    scaffs = {seed: [scaff_0]}
    for i in range(1, reinvent_rounds):
        path_i = f"{output_dir}/iteration{i}_{acquisition}/results/scaffold_memory.csv"
        scaff_i = pd.read_csv(path_i)
        scaffs[seed].append(scaff_i)
    return scaffs

def get_oracle_scores(df, score_component_name, top_mols):
    sorted_compounds = df.sort_values(by=[score_component_name], ascending=False)
    if top_mols:
        sorted_compounds = sorted_compounds.iloc[:top_mols]
    oracle_scores = [feedback_model.oracle_score(s) for s in sorted_compounds.SMILES] #replace with oracle from TDC benchmark
    return oracle_scores

def get_raw_scores(df, score_component_name, top_mols):
    sorted_compounds = df.sort_values(by=[score_component_name], ascending=False)
    if top_mols:
        sorted_compounds = sorted_compounds.iloc[:top_mols]
    raw_scores = sorted_compounds[f"raw_{score_component_name}"].values.tolist()
    return raw_scores

def get_average_score_by_iter(scaff_dict, score_component_name, reinvent_rounds, top_mols = None):
    avg_score_by_iter = dict()
    for iteration in range(1, reinvent_rounds):
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

def plot_oracle_score(
    baseline_output_dir, 
    compared_output_dir,  
    acquisition_method = "random",
    score_component_name = "bioactivity", 
    reinvent_rounds = 10,
    top_mols = None, 
    legend_drawn_flag = True,
    draw_molecules = False,
    save_to_path = "/home/klgx638/Generations/HITL_qsar_experiments/figures"
    ):

    scaffs_baseline = get_scaffold_memory_results(baseline_output_dir, reinvent_rounds)
    scaffs_compared = get_scaffold_memory_results(compared_output_dir, reinvent_rounds, acquisition_method)

    assert scaffs_baseline[1][0].total_score.values.mean() == scaffs_compared[1][0].total_score.values.mean()

    avg_score_by_iter_baseline = get_average_score_by_iter(scaffs_baseline, score_component_name, reinvent_rounds, top_mols)
    avg_score_by_iter_compared = get_average_score_by_iter(scaffs_compared, score_component_name, reinvent_rounds, top_mols)
    
    fig, ax = plt.subplots(1,2, figsize=(14,7))
    fig.tight_layout()

    #plot oracle score evolution
    means_by_iter_baseline = [np.mean(avg_score_by_iter_baseline[i]) for i in range(1, reinvent_rounds)]
    sd_by_iter_baseline = [np.std(avg_score_by_iter_baseline[i]) for i in range(1, reinvent_rounds)]
    
    means_by_iter_compared = [np.mean(avg_score_by_iter_compared[i]) for i in range(1, reinvent_rounds)]
    sd_by_iter_compared = [np.std(avg_score_by_iter_compared[i]) for i in range(1, reinvent_rounds)]

    ax[0].plot(np.arange(1, reinvent_rounds), means_by_iter_baseline)
    ax[0].plot(np.arange(1, reinvent_rounds), means_by_iter_compared)
    
    ax[0].fill_between(
            np.arange(1, reinvent_rounds), 
            np.array(means_by_iter_baseline) - np.array(sd_by_iter_baseline), 
            np.array(means_by_iter_baseline) + np.array(sd_by_iter_baseline), 
            alpha=.15
            )
    ax[0].fill_between(
            np.arange(1, reinvent_rounds), 
            np.array(means_by_iter_compared) - np.array(sd_by_iter_compared), 
            np.array(means_by_iter_compared) + np.array(sd_by_iter_compared), 
            alpha=.15
            )

    ax[0].set_xticks(np.arange(1, reinvent_rounds))
    ax[0].set_ylim(ymin = 0)
    ax[0].set_ylim(ymax = 1)
    ax[0].legend(["no human", f"{acquisition_method}"], loc=0, frameon=legend_drawn_flag)
    ax[0].set_ylabel("Oracle score")
    ax[0].set_xlabel("Policy optimization rounds (R)")
    ax[0].set_title("Oracle score evolution")

    #plot oracle scores distribution for last 100 rounds
    #for seed in scaffs_compared.keys():
    scaff_baseline = scaffs_baseline[1][-1]
    scaff_compared = scaffs_compared[1][-1]
    ax[1].hist(get_oracle_scores(scaff_baseline, score_component_name, top_mols), alpha=.3)
    ax[1].hist(get_oracle_scores(scaff_compared, score_component_name, top_mols), alpha=.3)
    ax[1].legend(["no human", f"{acquisition_method}"], loc=0, frameon=legend_drawn_flag)
    ax[1].set_ylabel("Frequency")
    ax[1].set_xlabel("Oracle score")
    ax[1].set_title(f"Oracle scores distrbution for last 100 rounds")

    fig.savefig(os.path.join(f"{save_to_path}", "results_ran.png"))

def plot_accuracy(
        baseline_output_dir, 
        compared_output_dir, 
        metric, 
        component_name = "bioactivity", 
        top_mols = 1000,
        reinvent_rounds = 10, 
        legend_drawn_flag = True,
        save_to_path = "/home/klgx638/Generations/HITL_qsar_experiments/figures"
        ):

        acquisition_method = compared_output_dir.split("/")[-1].split("_")[2]

        scaffs_baseline = get_scaffold_memory_results(baseline_output_dir, reinvent_rounds)
        scaffs_compared = get_scaffold_memory_results(compared_output_dir, reinvent_rounds, acquisition_method)

        r2_score_baseline = []
        r2_score_compared = []
        
        for i in range(1, reinvent_rounds):
            for seed in scaffs_baseline.keys():
                scaff_baseline = scaffs_baseline[seed][i]
                scaff_compared = scaffs_compared[seed][i]
                
                raw_values_baseline = get_raw_scores(scaff_baseline, component_name, top_mols)
                oracle_values_baseline = get_oracle_scores(scaff_baseline, component_name, top_mols)
                
                raw_values_compared = get_raw_scores(scaff_compared, component_name, top_mols)
                oracle_values_compared = get_oracle_scores(scaff_compared, component_name, top_mols)

            if metric == "r2": 
                r2_score_baseline.append(r2_score(oracle_values_baseline, raw_values_baseline))
                r2_score_compared.append(r2_score(oracle_values_compared, raw_values_compared))

                print(r2_score(oracle_values_baseline[:2], raw_values_baseline[:2]))
                print(r2_score(oracle_values_compared[:2], raw_values_compared[:2]))

        
        fig, ax = plt.subplots(figsize=(14,7))
        fig.tight_layout()
        ax.plot(np.arange(1, reinvent_rounds), r2_score_baseline)
        ax.plot(np.arange(1, reinvent_rounds), r2_score_compared)
        ax.legend(["no human", f"{acquisition_method}"], loc=0, frameon=legend_drawn_flag)
        ax.set_ylabel(f"{metric}")
        ax.set_xlabel("Iteration of human feedback")
        ax.set_title(f"Predictive model accuracy ({metric})")

        fig.savefig(os.path.join(f"{save_to_path}", f"accuracy_{acquisition_method}.png"))

        return fig    


if __name__ == "__main__":
    print(sys.argv)
    baseline_dir = str(sys.argv[1]) # acquisition: 'uncertainty', 'random', 'thompson', 'greedy'
    compared_dir = str(sys.argv[2])
    acquisition = str(sys.argv[3])
    metric = "r2"
    #plot_oracle_score(baseline_dir, compared_dir, acquisition)
    plot_accuracy(baseline_dir, compared_dir, metric)
