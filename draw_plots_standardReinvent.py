import pickle
import pandas as pd
import numpy as np

from rdkit import Chem
from utils import fingerprints_from_mol
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

# Load oracle model
mod = pickle.load(open("clf_p3.pkl", "rb"))

# Load initial model
initial_model = pickle.load(open("/home/klgx638/Projects/reinvent-hitl-calibration/bioactivity_models/drd2/rf_qsar_iteration_0.pkl", "rb"))

n_seeds = 10

all_exps = dict()
for seed in range(1,n_seeds+1):
    print(seed)
    path_nohuman = f"/home/klgx638/Generations/HITL_bioactivity/demo_rf_nohuman_seed{seed}/results/scaffold_memory.csv"
    scaff_nohuman = pd.read_csv(path_nohuman)

    fps0 = fingerprints_from_mol([Chem.MolFromSmiles(s) for s in scaff_nohuman.SMILES])
    scaff_nohuman["oracle"] = mod.predict_proba(fps0)[:,1].tolist()
    scaff_nohuman["model"] = initial_model.predict(fps0).tolist()
    scaff_nohuman["seed"] = [seed for i in range(len(scaff_nohuman))]
    all_exps[f"{seed}"] = scaff_nohuman

print("Calculating the means and stds")
dfs_to_concat = [all_exps[f"{seed}"] for seed in range(1,n_seeds+1)]
df_seed = pd.concat(dfs_to_concat)
df_grouped = df_seed[["Step", "seed", "bioactivity", "oracle", "model"]].groupby(["Step", "seed"])

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

ax.set_ylim(ymin = 0)
ax.set_xlim(xmin = 0)
ax.set_ylim(ymax = 1)
ax.set_xlim(xmax = 1000)

ax.set_title(f"Evolution of bioactivity score without expert fine-tuning")
ax.set_ylabel("Average bioactivity score")
ax.set_xlabel("REINVENT epochs")

fig.savefig(f"/home/klgx638/Projects/reinvent-hitl-calibration/figures/demo_rf_nohuman.png")