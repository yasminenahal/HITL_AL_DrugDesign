# load dependencies
import sys
import os
import shutil
import torch
import re
import json
import tempfile
import pandas as pd
import csv
from bioactivity_models import RandomForest
import numpy as np
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import math
import reinvent_scoring
from numpy.random import default_rng

from utils import fingerprints_from_mol
from scripts.simulated_expert_delta import ActivityEvaluationModel, logPEvaluationModel
from scripts.write_classif_logp import write_REINVENT_config
from bioactivity_models.RandomForest import RandomForestModel
from scripts.acquisition import select_query
#from scripts.REINVENTconfig import parse_config_file

#import pystan
#import stan_utility
import pickle


def do_run(acquisition, seed, rep):
    ##############################
    # Quick options
    FIT_MODEL = False # whether to fit a Stan model or not
    LOAD_MODEL = True # load Stan model from disc
    SUBSAMPLE = True # Reduce the size of the pool of unlabeled molecules to reduce computation time
    ##############################

    pure_random = False
    greedy = False
    unc = False
    
    if acquisition == "random":
        pure_random = True
        jobid = 'demo_rf_logp_classif_pureRandom_noise0.1'
    if acquisition == "greedy":
        greedy = True
        jobid = 'demo_rf_logp_classif_greedy_noise0.1'
    if acquisition == "qbc":
        unc = True
        jobid = 'demo_rf_logp_classif_qbc_noise0.1'
    jobname = "fine-tune logp model"

    np.random.seed(seed)
    rng = default_rng(seed)

    ########### HITL setup #################
    T = 1 # number of HITL iterations
    n = 30 # number of molecules shown to the simulated chemist at each iteration
    #n0 = 20 # number of molecules shown to the expert at initialization
    K = 10 # number of REINVENT runs: usage: K=2 for one HITL round (T*n queries); K=3 for two HITL rounds (2*(T*n) queries)
    ########################################
    
    bioactivity_model = 'rf_logp'
    bioactivity_identifier = bioactivity_model + '_' + jobid
    initial_model_path = f'/home/klgx638/Projects/reinvent-hitl-calibration/logp_model/{bioactivity_model}_classifier_24.pkl'

    # --------- change these path variables as required
    reinvent_dir = os.path.expanduser("/home/klgx638/Projects/reinventcli")
    reinvent_env = os.path.expanduser("/home/klgx638/miniconda3/envs/reinvent.v3.2") 
    output_dir = os.path.expanduser("/home/klgx638/Generations/HITL_logp/same_seed_trainonall_torch/{}_seed{}_rep{}".format(jobid, seed, rep))
    print("Running MPO experiment with K={}, T={}, n_queries={}, seed={}. \n Results will be saved at {}".format(K, T, n, seed, output_dir))

    # initialize human model
    feedback_model = logPEvaluationModel()
    print("Loading feedback model.")

    # load background training data for the bioactivity model
    #drd2_train = pd.read_csv("data/drd2/ECFP_counts_train_oracle.csv", index_col=0)
    train = pd.read_csv("data/logp/train_classif_24.csv", index_col=0)
    x_train = train.iloc[:,2:-2]
    y_train = train.Label.values
    sample_weight = np.array([1. for i in range(len(x_train))])
    print("Loading backrgound data.")
    print("Train features : ", x_train.shape)
    print("Train labels : ", y_train.shape)
    
    # initial configuration
    conf_filename = "config.json"

    # create root output dir
    if not os.path.exists(os.path.join(output_dir)):
        os.makedirs(output_dir)
    
    print(f"Creating output directory: {output_dir}.")
    configuration_JSON_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_dir, conf_filename, jobid, jobname)
    print(f"Creating config file: {configuration_JSON_path}.")

    # get the predictive model
    model_load_path = output_dir + '/{}_iteration_0.pkl'.format(bioactivity_model)
    if not os.path.exists(model_load_path):
        shutil.copy(initial_model_path, output_dir)
    fitted_model = pickle.load(open(initial_model_path, 'rb'))
    print("Loading predictive model.") 

    # store expert scores
    expert_score = []

    READ_ONLY = True # if folder exists, do not overwrite results there

    for REINVENT_iteration in np.arange(1,K+1):  

        torch.manual_seed(seed)

        #if (not READ_ONLY):
        print("RUN REINVENT")
        os.system(reinvent_env + '/bin/python ' + reinvent_dir + '/input.py ' + configuration_JSON_path + '&> ' + output_dir + '/run.err')
        #else:
        print("Reading REINVENT results from file.")
        with open(os.path.join(output_dir, "results/scaffold_memory.csv"), 'r') as file:
            data = pd.read_csv(file)
            
        N = len(data)
        colnames = list(data) 
        smiles = data['SMILES']
        bioactivity_score = data['bioactivity'] # the same as raw_bioactivity since no transformation applied
        raw_bioactivity_score = data['raw_bioactivity']
        high_scoring_threshold = 0.7
        # save the indexes of high scoring molecules for bioactivity
        high_scoring_idx = bioactivity_score >= high_scoring_threshold

        # Scoring component values
        # TODO: rename the prob_dists columns (remove the raw )
        scoring_component_names = [s.split("raw_")[1] for s in colnames if "raw_" in s]
        print(f"scoring components: {scoring_component_names}")
        x = np.array(data[scoring_component_names])
        print(f'Scoring component matrix dimensions: {x.shape}')
        x = x[high_scoring_idx,:]

        if greedy:
            # Only analyse highest scoring molecules
            smiles = smiles[high_scoring_idx]
            bioactivity_score = bioactivity_score[high_scoring_idx]
            raw_bioactivity_score = raw_bioactivity_score[high_scoring_idx]
            #print(f'{len(smiles)} molecules')
            print(f'{len(smiles)} high-scoring (> {high_scoring_threshold}) molecules')

            if len(smiles) == 0:
                smiles = data['SMILES']

        else:
            print(f'{len(smiles)} molecules')

        # mean and standard deviation of noise in expert feebback
        mu, sigma = 0, 0.1 # 
        
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
            if "rf" in bioactivity_model:
                fit = RandomForestModel(fitted_model)
            if "logreg" in bioactivity_model:
                fit = RandomForestModel(fitted_model["fit"])
            if len(smiles) > n:
                new_query = select_query(N, n, list(smiles), fit, selected_feedback, acquisition, rng) # select n smiles with AL
            else:
                new_query = select_query(N, len(smiles), list(smiles), fit, selected_feedback, acquisition, rng)
            # Initialize the expert values vector
            s_bioactivity = np.zeros(len(new_query))
            # Sample expert noise
            noise_sample = np.random.normal(mu, sigma, 1).item()
            #human_agreement_threshold = 1 + noise_sample
            human_range = [2 - noise_sample, 4 + noise_sample]
            # Get expert feedback on selected queries
            print(new_query)
            for i in np.arange(len(new_query)):
                print(list(smiles)[i])
                cur_mol = Chem.MolFromSmiles(list(smiles)[i])
                # Reward score for query molecule i
                raw_score = raw_bioactivity_score.iloc[i]
                s_bioactivity[i] = feedback_model.feedback_label(cur_mol, noise_sample, human_range)
                #s_bioactivity[i] = feedback_model.feedback_score(cur_mol, noise_sample, raw_score, human_agreement_threshold)
                #s_bioactivity[i] = feedback_model.feedback_score(cur_mol, noise_sample)
                #except:
                #print("INVALID MOLECULE in scaffold memory")
                #s_bioactivity[i] = 0
            
            # Get raw scores and transformed score (if any) from the high scoring molecules in U
            raw_scoring_component_names = ["raw_"+name for name in scoring_component_names] 
            x_raw = data[raw_scoring_component_names].to_numpy()
            x =  data[scoring_component_names].to_numpy()

            # get (binary) simulated chemist's responses
            
            #new_y = [1 if s > 0.5 else 0 for s in s_bioactivity]
            new_y = [int(s) for s in s_bioactivity]
            print(new_y)
            #n_accept += [sum(new_y)]

            print(f"Feedback idx at iteration {REINVENT_iteration}, {t}: {new_query}")
            print(f"Number of accepted molecules at iteration {REINVENT_iteration}, {t}: {new_y.count(1)}")
            
            new_y_tokeep = new_y
            new_query_tokeep = new_query
            
            print(len(new_y_tokeep))
            print(len(new_query_tokeep))

            expert_score += [new_y_tokeep]
            
            # append feedback
            if len(new_y_tokeep) > 0:
                selected_feedback = np.hstack((selected_feedback, new_query_tokeep))
                y = np.hstack((y, new_y_tokeep))

            mask = np.ones(N, dtype=bool)
            mask[selected_feedback] = False

        # use the augmented training data to retrain the model
        new_mols = [Chem.MolFromSmiles(s) for s in data.iloc[selected_feedback].SMILES.tolist()]
        new_x = fingerprints_from_mol(new_mols, type = "counts")
        new_human_sample_weight = np.array([2. for i in range(len(new_x))])
        human_sample_weight = np.concatenate([human_sample_weight, new_human_sample_weight])
        sample_weight = np.concatenate([sample_weight, human_sample_weight])
        x_train = np.concatenate([x_train, new_x])
        y_train = np.hstack((y_train, y))
        print(f"Augmented train set size at iteration {REINVENT_iteration}: {x_train.shape[0]}")

        # get current configuration
        configuration = json.load(open(os.path.join(output_dir, conf_filename)))
        conf_filename = "iteration{}_config.json".format(REINVENT_iteration)

        # Keep agent checkpoint
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_dir, "results/Agent.ckpt")
        root_output_dir = os.path.expanduser("/home/klgx638/Generations/HITL_logp/same_seed_trainonall_torch/{}_seed{}_rep{}".format(jobid, seed, rep))

        # Define new directory for the next round
        output_dir = os.path.join(root_output_dir, "iteration{}_{}".format(REINVENT_iteration, acquisition))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(output_dir)

        # re-fit and save the model using the augmented train set and save to new directory
        model_new_savefile = output_dir + '/{}_iteration_{}.pkl'.format(bioactivity_model, REINVENT_iteration)
        fit._reinitialize_classifier(x_train, y_train, sample_weight = sample_weight, save_to_path = model_new_savefile)       

        # modify model path in configuration
        configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
        for i in range(len(configuration_scoring_function)):
            if configuration_scoring_function[i]["component_type"] == "predictive_property":
                configuration_scoring_function[i]["specific_parameters"]["model_path"] = model_new_savefile

        # modify log and result paths in configuration
        configuration["logging"]["logging_path"] = os.path.join(output_dir, "progress.log")
        configuration["logging"]["result_folder"] = os.path.join(output_dir, "results")

        # write the updated configuration file to the disc
        configuration_JSON_path = os.path.join(output_dir, conf_filename)
        with open(configuration_JSON_path, 'w') as f:
            json.dump(configuration, f, indent=4, sort_keys=True)

        # Run REINVENT again
        #READ_ONLY = False
    

if __name__ == "__main__":
    print(sys.argv)
    acquisition = sys.argv[1] # acquisition: 'uncertainty', 'random', 'thompson', 'greedy'
    seed = int(sys.argv[2])
    rep = int(sys.argv[3])
    do_run(acquisition, seed, rep)
