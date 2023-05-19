import pickle

import pandas as pd
import numpy as np
from data_preprocessing import MorganECFPCounts
from utils import fingerprints_from_mol
from rdkit import Chem
from rdkit.Chem.Crippen import MolLogP
from matplotlib import pyplot as plt

train = pd.read_csv("/home/klgx638/Projects/reinvent-hitl-calibration/data/logp/train_classif_24.csv")
test = pd.read_csv("/home/klgx638/Projects/reinvent-hitl-calibration/data/logp/test_classif_24.csv")

def get_scaffold_memory_results(objective, setting, seed, rep, sampling=None, noise=0.1):
    if objective == "logp":
        output_dir = "/home/klgx638/Generations/HITL_logp"
    elif objective == "drd2":
        output_dir = "/home/klgx638/Generations/HITL_bioactivity/same_seed/"
    if sampling is not None:
        path0 = f"{output_dir}/demo_rf_{objective}_{setting}_nohuman_rep{rep}/results/scaffold_memory.csv"
        scaff = pd.read_csv(path0)
        scaff_0 = [scaff[scaff["Step"]<100]]

        scaffs = []
        for i in range(1,10):
            if sampling == "pureRandom":
                itr = f"{output_dir}/same_seed_trainonall_torch/demo_rf_{objective}_{setting}_{sampling}_noise{noise}_seed{seed}_rep{rep}/iteration{i}_random/results/scaffold_memory.csv"
            else:
                itr = f"{output_dir}/same_seed_trainonall_torch/demo_rf_{objective}_{setting}_{sampling}_noise{noise}_seed{seed}_rep{rep}/iteration{i}_{sampling}/results/scaffold_memory.csv"
            scaff2 = pd.read_csv(itr)
            scaffs.append(scaff2)

        scaffs = scaff_0 + scaffs
        return scaffs
    else:
        path = f"{output_dir}/demo_rf_{objective}_{setting}_nohuman_rep{rep}/results/scaffold_memory.csv"
        scaff = pd.read_csv(path)
        return scaff

from rdkit.Chem.Crippen import MolLogP

#LOGP
all_tps_nohuman = []
all_tps_greedy_onerep = []
all_tps_pureRandom_onerep = []
all_tps_qbc_onerep = []
s = 22222
for iterr in range(10):
    tps = []
    for i in range(1,11):
        scaff = get_scaffold_memory_results("logp", "classif", s, i, "greedy")[iterr]
        high_scoring_compounds = scaff[scaff["bioactivity"] > 0.7]
        if high_scoring_compounds.shape[0] > 0:
            sorted_high_scoring_compounds = high_scoring_compounds.sort_values(by=['bioactivity'], ascending=False)
            sorted_high_scoring_compounds_top = sorted_high_scoring_compounds.iloc[:3000]
            high_scoring_mols = [Chem.MolFromSmiles(s) for s in sorted_high_scoring_compounds_top.SMILES]
            sorted_high_scoring_compounds_top["oracle"] = [1 if MolLogP(m) >= 2 and MolLogP(m) <= 4 else 0 for m in high_scoring_mols]
            tp = sorted_high_scoring_compounds_top["oracle"].tolist().count(1) / sorted_high_scoring_compounds_top.shape[0]
        else:
            tp = 0
        tps.append(tp)
    all_tps_greedy_onerep.append(tps)
for iterr in range(10):
    tps = []
    for i in range(1,11):
        scaff = get_scaffold_memory_results("logp", "classif", s, i, "pureRandom")[iterr]
        high_scoring_compounds = scaff[scaff["bioactivity"] > 0.7]
        if high_scoring_compounds.shape[0] > 0:
            sorted_high_scoring_compounds = high_scoring_compounds.sort_values(by=['bioactivity'], ascending=False)
            sorted_high_scoring_compounds_top = sorted_high_scoring_compounds.iloc[:3000]
            high_scoring_mols = [Chem.MolFromSmiles(s) for s in sorted_high_scoring_compounds_top.SMILES]
            sorted_high_scoring_compounds_top["oracle"] = [1 if MolLogP(m) >= 2 and MolLogP(m) <= 4 else 0 for m in high_scoring_mols]
            tp = sorted_high_scoring_compounds_top["oracle"].tolist().count(1) / sorted_high_scoring_compounds_top.shape[0]
        else:
            tp = 0
        tps.append(tp)
    all_tps_pureRandom_onerep.append(tps)
for iterr in range(10):
    tps = []
    for i in range(1,11):
        scaff = get_scaffold_memory_results("logp", "classif", s, i, "qbc")[iterr]
        high_scoring_compounds = scaff[scaff["bioactivity"] > 0.7]
        if high_scoring_compounds.shape[0] > 0:
            sorted_high_scoring_compounds = high_scoring_compounds.sort_values(by=['bioactivity'], ascending=False)
            sorted_high_scoring_compounds_top = sorted_high_scoring_compounds.iloc[:3000]
            high_scoring_mols = [Chem.MolFromSmiles(s) for s in sorted_high_scoring_compounds_top.SMILES]
            sorted_high_scoring_compounds_top["oracle"] = [1 if MolLogP(m) >= 2 and MolLogP(m) <= 4 else 0 for m in high_scoring_mols]
            tp = sorted_high_scoring_compounds_top["oracle"].tolist().count(1) / sorted_high_scoring_compounds_top.shape[0]
        else:
            tp = 0
        tps.append(tp)
    all_tps_qbc_onerep.append(tps)
j1 = 0
j2 = 100
for iterr in range(10):
    tps = []
    for i in range(1,11):
        scaff = get_scaffold_memory_results("logp", "classif", s, i)
        scaff = scaff[scaff["Step"]>j1]
        scaff = scaff[scaff["Step"]<j2]
        high_scoring_compounds = scaff[scaff["bioactivity"] > 0.7]
        if high_scoring_compounds.shape[0] > 0:
            sorted_high_scoring_compounds = high_scoring_compounds.sort_values(by=['bioactivity'], ascending=False)
            sorted_high_scoring_compounds_top = sorted_high_scoring_compounds.iloc[:3000]
            high_scoring_mols = [Chem.MolFromSmiles(s) for s in sorted_high_scoring_compounds_top.SMILES]
            sorted_high_scoring_compounds_top["oracle"] = [1 if MolLogP(m) >= 2 and MolLogP(m) <= 4 else 0 for m in high_scoring_mols]
            tp = sorted_high_scoring_compounds_top["oracle"].tolist().count(1) / sorted_high_scoring_compounds_top.shape[0]
        else:
            tp = 0
        tps.append(tp)
    all_tps_nohuman.append(tps)
    j1 += 100
    j2 += 100

def plotly(liste, metric = "mean"):
    y_lower = []
    y_upper = []
    means = []
    for i in range(len(liste)):
        means.append(np.mean(liste[i]))
        y_lower.append(np.mean(liste[i]) - np.std(liste[i]))
        y_upper.append(np.mean(liste[i]) + np.std(liste[i]))
    if metric == "mean":
        return means
    if metric == "lower":
        return y_lower
    if metric == "upper":
        return y_upper

means = [plotly(l, "mean") for l in [all_tps_nohuman, all_tps_greedy_onerep, all_tps_pureRandom_onerep, all_tps_qbc_onerep]]
y_lower = [plotly(l, "lower") for l in [all_tps_nohuman, all_tps_greedy_onerep, all_tps_pureRandom_onerep, all_tps_qbc_onerep]]
y_upper = [plotly(l, "upper") for l in [all_tps_nohuman, all_tps_greedy_onerep, all_tps_pureRandom_onerep, all_tps_qbc_onerep]]
fig, ax = plt.subplots()

for i in range(len(means)):
    ax.plot(np.arange(10), means[i])
legend_drawn_flag = True
ax.legend(["No HITL", "HITL + Greedy sampling", "HITL + random sampling", "HITL + QBC sampling"], loc=0, frameon=legend_drawn_flag)
for i in range(len(means)):
    ax.fill_between(
        np.arange(10), y_lower[i], y_upper[i], alpha=.15)
ax.set_xticks(np.arange(10))
ax.set_ylim(ymin = 0)
ax.set_ylim(ymax = 1)
ax.set_ylabel("True positive rate \n (Crippen logP in [2,4])")
ax.set_xlabel("HITL iterations")
ax.set_title("REINVENT optimization for oral drug absorption (logP in [2,4])")

fig.savefig(f"/home/klgx638/Projects/reinvent-hitl-calibration/figures/logp_differentReps_top3000_torch.png")