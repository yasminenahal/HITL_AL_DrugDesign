import sys
import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from utils import fingerprints_from_mol
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

print(sys.argv)
experiment = sys.argv[1]
if "qbc" in experiment:
    acq = "qbc"
if "random" in experiment:
    acq = "random"
if "noise" in experiment:
    b = experiment.split("_sigmoid",1)[0]
    noise = float(b.split("noise",1)[1])
else:
    noise = 0.1

n_seeds = 10

# Load oracle model
mod = pickle.load(open("/home/klgx638/Projects/reinvent-hitl-calibration/clf_p3.pkl", "rb"))

# Get all the scores and draw the plots

all_exps = dict()
for seed in range(1,n_seeds+1):
    print(seed)
    path0 = f"/home/klgx638/Generations/HITL_bioactivity/{experiment}_seed{seed}/results/scaffold_memory.csv"
    first_scaff = pd.read_csv(path0).sort_values(by = "Step")
    first_scaff_fps = fingerprints_from_mol([Chem.MolFromSmiles(s) for s in first_scaff.SMILES])
    first_scaff["oracle"] = mod.predict_proba(first_scaff_fps)[:,1].tolist()
    first_scaff["oracleLabel"] = mod.predict(first_scaff_fps).tolist()
    initial_model = pickle.load(open("/home/klgx638/Projects/reinvent-hitl-calibration/bioactivity_models/drd2/rf_qsar_iteration_0.pkl", "rb"))
    first_scaff["model"] = initial_model.predict(first_scaff_fps).tolist()
    first_scaff["model0"] = initial_model.predict(first_scaff_fps).tolist()
    first_scaff["seed"] = [seed for i in range(len(first_scaff))]
    scaffs = [first_scaff]
    for i in range(1,10):
        #print(i)
        path = f"/home/klgx638/Generations/HITL_bioactivity/{experiment}_seed{seed}/iteration{i}_{acq}/results/scaffold_memory.csv"
        scaff = pd.read_csv(path).sort_values(by = "Step")
        scaff_fps = fingerprints_from_mol([Chem.MolFromSmiles(s) for s in scaff.SMILES])
        scaff["oracle"] = mod.predict_proba(scaff_fps)[:,1].tolist()
        scaff["oracleLabel"] = mod.predict(scaff_fps).tolist()
        updated_model = pickle.load(open(f"/home/klgx638/Generations/HITL_bioactivity/{experiment}_seed{seed}/iteration{i}_{acq}/rf_qsar_iteration_{i}.pkl", "rb"))
        scaff["model"] = updated_model.predict(scaff_fps).tolist()
        scaff["model0"] = initial_model.predict(scaff_fps).tolist()
        scaff["seed"] = [seed for i in range(len(scaff))]
        #print(len(scaff_fps), len(scaff.oracle))
        scaffs.append(scaff)
    all_exps[f"{seed}"] = scaffs

new_step_col = []
for seed in range(1,n_seeds+1):
    print(seed)
    all_exps[f"{seed}"][0]["Step_"] = all_exps[f"{seed}"][0]["Step"].tolist()
    v = 100
    for j in range(1, len(all_exps[f"{seed}"])):
        new_step_col = []
        for _, row in all_exps[f"{seed}"][j].iterrows():
            new_step_col.append(row.Step + v)
        all_exps[f"{seed}"][j]["Step_"] = new_step_col
        v += 100

print("Calculating the means and stds")
dfs_to_concat = []
for seed in range(1,n_seeds+1):
    dfs_to_concat += all_exps[f"{seed}"]
df_seed = pd.concat(dfs_to_concat)
df_grouped = df_seed[["Step_", "seed", "bioactivity", "oracle", "model", "model0"]].groupby(["Step_", "seed"])

df_grouped_bioactivity = df_grouped["bioactivity"].agg(["mean", "std"]).fillna(0)
df_grouped_oracle = df_grouped["oracle"].agg(["mean", "std"]).fillna(0)

df_grouped_bioactivity["upper"] = df_grouped_bioactivity["mean"] + df_grouped_bioactivity["std"]
df_grouped_bioactivity["lower"] = df_grouped_bioactivity["mean"] - df_grouped_bioactivity["std"]
df_grouped_oracle["upper"] = df_grouped_oracle["mean"] + df_grouped_oracle["std"]
df_grouped_oracle["lower"] = df_grouped_oracle["mean"] - df_grouped_oracle["std"]

steps = [t[0] for t in df_grouped_bioactivity.index.to_list()]

print("Plotting")
x = np.array(sorted(steps))
y_1 = df_grouped_bioactivity["mean"]
y_2 = df_grouped_oracle["mean"]
y_1_lower = df_grouped_bioactivity["lower"]
y_1_upper = df_grouped_bioactivity["upper"]
y_2_lower = df_grouped_oracle["lower"]
y_2_upper = df_grouped_oracle["upper"]
X_Y_Spline_1 = interp1d(x, y_1)
X_Y_Spline_2 = interp1d(x, y_2)

X_Y_1_lower = interp1d(x, y_1_lower)
X_Y_1_upper = interp1d(x, y_1_upper)
X_Y_2_lower = interp1d(x, y_2_lower)
X_Y_2_upper = interp1d(x, y_2_upper)
 
# Returns evenly spaced numbers
# over a specified interval.
X_ = np.linspace(x.min(), x.max(), 100)
Y_1 = X_Y_Spline_1(X_)
Y_2 = X_Y_Spline_2(X_)
Y_1_upper = X_Y_1_upper(X_)
Y_1_lower = X_Y_1_lower(X_)
Y_2_upper = X_Y_2_upper(X_)
Y_2_lower = X_Y_2_lower(X_)

fig, ax = plt.subplots()

ax.plot(X_, Y_1)
ax.plot(X_, Y_2)

legend_drawn_flag = True
ax.legend(["Optimized score", "Oracle score"], loc=0, frameon=legend_drawn_flag)

ax.fill_between(
    X_, Y_1_lower, Y_1_upper, color='blue', alpha=.15)
ax.fill_between(
    X_, Y_2_lower, Y_2_upper, color='orange', alpha=.15)

ax.axhline(y = 0.5, color = 'red', linestyle = '-')
for hitl in range(100,1000,100):
    ax.axvline(x = hitl, color = 'green', linestyle = '--')

ax.set_ylim(ymin = 0)
ax.set_xlim(xmin = 0)
ax.set_ylim(ymax = 1)
ax.set_xlim(xmax = 1000)

ax.set_title(f"Evolution of bioactivity score with expert fine-tuning \n {acq} queries selection, expert noise = {noise}")
ax.set_ylabel("Average bioactivity score")
ax.set_xlabel("REINVENT epochs")

fig.savefig(f"/home/klgx638/Projects/reinvent-hitl-calibration/figures/{experiment}.png")