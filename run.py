# load dependencies
import sys
import pickle
import os
import shutil
import json
import pandas as pd
import numpy as np
import rdkit.Chem as Chem
from numpy.random import default_rng

from utils import fingerprints_from_mol
from scripts.simulated_expert_delta import ActivityEvaluationModel, logPEvaluationModel
from scripts.write_classif_logp import write_REINVENT_config
from bioactivity_models.RandomForest import RandomForestReg
from scripts.acquisition import select_query

def do_run(seed, acquisition = None, sigma_noise = 0):
    ##############################
    # Quick options
    FIT_MODEL = False # whether to fit a Stan model or not
    LOAD_MODEL = True # load Stan model from disc
    SUBSAMPLE = True # Reduce the size of the pool of unlabeled molecules to reduce computation time
    ##############################
    
    jobid = f"demo_logpSmall_{acquisition}_noise{sigma_noise}"
    jobname = "fine-tune predictive component"

    np.random.seed(seed)
    rng = default_rng(seed)

    ########### HITL setup #################
    T = 1 # number of HITL iterations
    n = 30 # number of molecules shown to the simulated chemist at each iteration
    #n0 = 20 # number of molecules shown to the expert at initialization
    K = 10 # number of REINVENT runs: usage: K=2 for one HITL round (T*n queries); K=3 for two HITL rounds (2*(T*n) queries)
    ########################################
    
    bioactivity_model = "toy_example_OracleBased"
    initial_model_path = f"/home/klgx638/Projects/reinvent-hitl-calibration/logp_model/{bioactivity_model}.pkl"

    # change these path variables as required
    reinvent_dir = os.path.expanduser("/home/klgx638/Projects/reinventcli")
    reinvent_env = os.path.expanduser("/home/klgx638/miniconda3/envs/reinvent.v3.2-updated")
    output_dir = os.path.expanduser(f"/home/klgx638/Generations/HITL_qsar_experiments/{jobid}_seed{seed}")
    
    # initial configuration
    conf_filename = "config.json"

    # create root output dir
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    
    print(f"Creating output directory: {output_dir}.")
    configuration_JSON_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, conf_filename, jobid, jobname)
    print(f"Creating config file: {configuration_JSON_path}.")

    if acquisition:
        # human interaction starts from the same first batch of generated compounds
        initial_dir = f"/home/klgx638/Generations/HITL_qsar_experiments/demo_logpSmall_None_noise{sigma_noise}_seed{seed}"
        # create an initialization folder
        os.makedirs(os.path.join(output_dir, "iteration_0"))
        # copy initialization scaffold memory in the initialization folder
        shutil.copy(os.path.join(initial_dir, "results/scaffold_memory.csv"), os.path.join(output_dir, "iteration_0"))
    
    print(f"Running MPO experiment with K={K}, T={T}, n_queries={n}, seed={seed}. \n Results will be saved at {output_dir}")

    # initialize human feedback model
    feedback_model = logPEvaluationModel()
    print("Loading feedback model.")

    # load background training data used to pre-train the predictive model
    print("Loading D0.")
    train_set = pd.read_csv("data/logp/logp_train_ECFP_counts.csv")
    feature_cols = [f"bit{i}" for i in range(2048)]
    target_col = ["activity"] #logp values
    x_train = train_set[feature_cols].values
    y_train = train_set[target_col].values.reshape(-1)
    sample_weight = np.array([1. for i in range(len(x_train))])
    print("Feature matrix : ", x_train.shape)
    print("Labels : ", y_train.shape)

    # load the predictive model
    model_load_path = output_dir + '/{}_iteration_0.pkl'.format(bioactivity_model)
    if not os.path.exists(model_load_path):
        shutil.copy(initial_model_path, output_dir)
    fitted_model = pickle.load(open(initial_model_path, 'rb'))
    print("Loading predictive model.")

    # store expert scores
    expert_score = []

    READ_ONLY = False # if folder exists, do not overwrite results there

    for REINVENT_iteration in np.arange(1,K+1):

        if REINVENT_iteration == 1 and acquisition: # skip REINVENT run and start from generated compounds in initial dir
            with open(os.path.join(output_dir, "iteration_0/scaffold_memory.csv"), 'r') as file:
                data = pd.read_csv(file)
            data = data[data["Step"] < 100]
            data.reset_index(inplace=True)

        else:
            if(not READ_ONLY):
                # run REINVENT
                print("Run REINVENT")
                os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir + '/run.err')
            else:
                print("Reading REINVENT results from file, no re-running.")
                pass

            with open(os.path.join(output_dir, "results/scaffold_memory.csv"), 'r') as file:
                data = pd.read_csv(file)
            
        N = len(data)
        colnames = list(data) 
        smiles = data['SMILES']
        bioactivity_score = data['bioactivity'] # the same as raw_bioactivity since no transformation applied
        raw_bioactivity_score = data['raw_bioactivity']
        high_scoring_threshold = 0.5
        # save the indexes of high scoring molecules for bioactivity
        high_scoring_idx = bioactivity_score > high_scoring_threshold

        # Scoring component values
        scoring_component_names = [s.split("raw_")[1] for s in colnames if "raw_" in s]
        print(f"scoring components: {scoring_component_names}")
        x = np.array(data[scoring_component_names])
        print(f'Scoring component matrix dimensions: {x.shape}')
        x = x[high_scoring_idx,:]

        # Only analyse highest scoring molecules
        smiles = smiles[high_scoring_idx]
        bioactivity_score = bioactivity_score[high_scoring_idx]
        raw_bioactivity_score = raw_bioactivity_score[high_scoring_idx]
        print(f'{len(smiles)} high-scoring (> {high_scoring_threshold}) molecules')

        if len(smiles) == 0:
            smiles = data['SMILES']
            print(f'{len(smiles)} molecules')

        if acquisition:
            # mean and standard deviation of noise in expert feebback
            mu, sigma = 0, sigma_noise # 
            
            # store molecule indexes selected for feedback
            selected_feedback = np.empty(0).astype(int)
            human_sample_weight = np.empty(0).astype(float)
            # store expert binary responses
            y = []
            # store number of accepted queries (y = 1) at each iteration
            n_accept = []

            ########################### HITL rounds ######################################
            for t in np.arange(T): # T number of HITL iterations
                print(f"iteration k={REINVENT_iteration}, t={t}")
                # query selection
                fit = RandomForestReg(fitted_model) # create function to load predictive model
                if len(smiles) > n:
                    new_query = select_query(N, n, list(smiles), fit, selected_feedback, acquisition, rng) # select n smiles with AL
                else:
                    new_query = select_query(N, len(smiles), list(smiles), fit, selected_feedback, acquisition, rng)
                # Initialize the expert values vector
                s_bioactivity = np.zeros(len(new_query))
                v_bioactivity = np.zeros(len(new_query))
                # Sample expert noise
                noise_sample = np.random.normal(mu, sigma, 1).item()
                # Get expert feedback on selected queries
                print(new_query)
                for i in np.arange(len(new_query)):
                    print(list(smiles)[i])
                    #cur_mol = Chem.MolFromSmiles(list(smiles)[i])
                    cur_mol = list(smiles)[i]
                    v_bioactivity[i] = feedback_model.human_score(cur_mol, noise_sample)
                    s_bioactivity[i] = feedback_model.utility(v_bioactivity[i], low = 2, high = 4)
                
                # Get raw scores and transformed score (if any) from the high scoring molecules in U
                raw_scoring_component_names = ["raw_"+name for name in scoring_component_names] 
                x_raw = data[raw_scoring_component_names].to_numpy()
                x =  data[scoring_component_names].to_numpy()

                # get (binary) simulated chemist's responses
                new_y = v_bioactivity
                accepted = [1 if s > 0.5 else 0 for s in s_bioactivity]
                n_accept += [sum(accepted)]

                print(f"Feedback idx at iteration {REINVENT_iteration}, {t}: {new_query}")
                print(f"Number of accepted molecules at iteration {REINVENT_iteration}, {t}: {n_accept[t]}")

                expert_score += [s_bioactivity]
                
                # append feedback
                if len(new_y) > 0:
                    selected_feedback = np.hstack((selected_feedback, new_query))
                    y = np.concatenate([y, new_y])

                mask = np.ones(N, dtype=bool)
                mask[selected_feedback] = False

            # use the augmented training data to retrain the model
            new_mols = [Chem.MolFromSmiles(s) for s in data.iloc[selected_feedback].SMILES.tolist()]
            new_x = fingerprints_from_mol(new_mols, type = "counts")
            new_human_sample_weight = np.array([s if s > 0.5 else 1-s for s in s_bioactivity])
            human_sample_weight = np.concatenate([human_sample_weight, new_human_sample_weight])
            sample_weight = np.concatenate([sample_weight, human_sample_weight])
            x_train = np.concatenate([x_train, new_x])
            y_train = np.concatenate([y_train, y])
            print(f"Augmented train set size at iteration {REINVENT_iteration}: {x_train.shape[0]} {y_train.shape[0]}")

            # re-fit and save the model using the augmented train set and save to new directory
            model_new_savefile = output_dir + '/{}_iteration_{}.pkl'.format(bioactivity_model, REINVENT_iteration)
            fit._reinitialize_regressor(x_train, y_train, sample_weight = sample_weight, save_to_path = model_new_savefile)   

            # get current configuration
            configuration = json.load(open(os.path.join(output_dir, conf_filename)))
            conf_filename = "iteration{}_config.json".format(REINVENT_iteration)    

            # modify model path in configuration
            configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
            for i in range(len(configuration_scoring_function)):
                if configuration_scoring_function[i]["component_type"] == "predictive_property":
                    configuration_scoring_function[i]["specific_parameters"]["model_path"] = model_new_savefile

            # Keep agent checkpoint
            if REINVENT_iteration == 1:
                configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(initial_dir, "results/Agent.ckpt")
            else:
                configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_dir, "results/Agent.ckpt")
        
        else:
            # get current configuration
            configuration = json.load(open(os.path.join(output_dir, conf_filename)))
            conf_filename = "iteration{}_config.json".format(REINVENT_iteration) 
            configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_dir, "results/Agent.ckpt")

        root_output_dir = os.path.expanduser("/home/klgx638/Generations/HITL_qsar_experiments/{}_seed{}".format(jobid, seed))

        # Define new directory for the next round
        output_dir = os.path.join(root_output_dir, "iteration{}_{}".format(REINVENT_iteration, acquisition))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)

        # modify log and result paths in configuration
        configuration["logging"]["logging_path"] = os.path.join(output_dir, "progress.log")
        configuration["logging"]["result_folder"] = os.path.join(output_dir, "results")

        # write the updated configuration file to the disc
        configuration_JSON_path = os.path.join(output_dir, conf_filename)
        with open(configuration_JSON_path, 'w') as f:
            json.dump(configuration, f, indent=4, sort_keys=True)

    r = np.arange(len(expert_score))
    m_score = [np.mean(expert_score[i]) for i in r]
    print("Mean expert score : ", m_score)

if __name__ == "__main__":
    print(sys.argv)
    seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        acquisition = sys.argv[2] # acquisition: 'uncertainty', 'random', 'thompson', 'greedy' (if None run with no human interaction)
        sigma_noise = int(sys.argv[3])
        do_run(seed, acquisition, sigma_noise)
    else:
        do_run(seed)
