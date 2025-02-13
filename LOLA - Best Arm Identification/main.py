import pandas as pd
import numpy as np
from arm_identify import LOLA
import progressbar
import matplotlib.pyplot as plt
import pickle
from plot import plot
import os

# load calibrate data

calibrate = pd.read_csv('original/Yufeng_CTR_calibrate.csv')


result_dir = 'tmp_results'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

df = pd.read_csv('original/Yufeng_CTR_test.csv')
# rows with the same test_id are different headlines for the same news
# ini_CTR is the predicted CTR for each headline, using Zikun's prediction model based on text embedding


# initial_pulls_ls = [20, 50, 100, 200, 300, 400, 500, 700, 1000]
initial_pulls_ls = [300]
c_Phi =  0.0015

print('--------------------------')
# print(initial_pulls_, c_Phi)
epsilon = 0 # gap between the best arm and allowed arms
#delta = 0.05 # faliure probability
delta = 0.2 # faliure probability

gamma = 0.005
RATIO = 1 # how much dataset to use

def run_one_exp(df, c_Phi, initial_pulls_, test_id_ls, INIT_WITH_PRED, METHOD, MODE, MAX_PULLS, VERBOSE, RANDOM_PULLING, COMPARE_PRED, A_B_TEST):
    # run on all news
    total_pulling_rec = []
    n_empty_result = 0

    n_success = 0
    _ = 0

    for test_id in progressbar.progressbar(test_id_ls):
        _ += 1
        one_instance = df[df['test_id'] == test_id]
        lola = LOLA(real_CTR=one_instance['CTR'].values, predicted_CTR=one_instance['ini_CTR'].values, 
                    epsilon=epsilon, delta=delta, gamma=gamma, c_Phi=c_Phi, 
                    MODE=MODE, init_with_pred=INIT_WITH_PRED, initial_pulls=initial_pulls_, 
                    max_pulls=MAX_PULLS, verbose=VERBOSE, random_pulling=RANDOM_PULLING, 
                    compare_pred=COMPARE_PRED, a_b_test=A_B_TEST)
        selected_arm, oracle_arm, n_arm, n_pulling, success = lola.run()

        if len(selected_arm) == 0:
            n_empty_result += 1


        if success:
            n_success += 1
        total_pulling_rec.append(n_pulling)

        if _ % 100 == 0:
            plt.hist(total_pulling_rec)
            plt.title('Average pulling: {}, Success rate: {}'.format(np.mean(total_pulling_rec), n_success/_))
            plt.savefig(os.path.join(result_dir, 'Method_{} total_pulling_rec_USE_PREDICTED_{}.png'.format(METHOD, INIT_WITH_PRED)))
            plt.close()

    print('average pulling: {}'.format(np.mean(total_pulling_rec)))
    print('success rate: {}'.format(n_success/len(test_id_ls)))
    print('empty result ratio: {}'.format(n_empty_result/len(test_id_ls)))
    # save the result using pkl
    return total_pulling_rec, n_success


def run_exp_LLM():
    # load data/result.csv
    # data/result.csv is the result of the prediction model (Finetune LLama with LoRA)
    # Copied from Release/24-7-19 Finetune CTR Prediction/20240719-174532/result.csv
    data_with_pred_CTR = pd.read_csv('data/result.csv')
    # assert the headlines in data_with_pred_CTR are the same as in df
    assert len(data_with_pred_CTR) == len(df)
    assert all(data_with_pred_CTR['headline'] == df['headline'])
    # for each news (headlines with same test_id), select the headline with the highest predicted CTR
    test_id_ls = df['test_id'].unique()
    n_test_id = int(len(test_id_ls) * RATIO)
    test_id_ls = test_id_ls[:n_test_id]
    n_success = 0

    for test_id in test_id_ls:
        one_instance = data_with_pred_CTR[data_with_pred_CTR['test_id'] == test_id]

        lola = LOLA(real_CTR=one_instance['real_CTR'].values, predicted_CTR=None, 
        epsilon=epsilon, delta=delta, gamma=gamma, c_Phi=c_Phi, 
        MODE='add', init_with_pred=None, initial_pulls=None, 
        max_pulls=None, verbose=None, random_pulling=None, 
        compare_pred=None, a_b_test=None)

        selected_arm = [np.argmax(one_instance['predictions'].values)]

        if lola.check_success(selected_arm):
            n_success += 1
        
        


        # real_best_headline = one_instance['real_CTR'].idxmax()
        # pred_best_headline = one_instance['predictions'].idxmax()
        # if real_best_headline == pred_best_headline:
        #     n_success += 1
    return n_success



def run_all_exp(df, c_Phi, initial_pulls_ls, filename):
    recording = {} # recording pulling number and success rate for each method
    METHOD_LS = ['LLM', 'LOLA', 'ST2', 'A_B_200', 'A_B_300', 'A_B_400', 'A_B_500', 'A_B_600'] # 'DEBUG'
    # METHOD_LS = ['PREDICTION']
    for METHOD in METHOD_LS:
        print('Method: {}'.format(METHOD))

        test_id_ls = df['test_id'].unique()
        n_test_id = int(len(test_id_ls) * RATIO)
        test_id_ls = test_id_ls[:n_test_id]


        max_min_CTR_gap_ls = []
        for test_id in test_id_ls:
            one_instance = df[df['test_id'] == test_id]
            max_min_CTR_gap_ls.append(max(one_instance['CTR']) - min(one_instance['CTR']))

        # LOLA: ST2 using predicted CTR as initialization
        # ST2: ST2 without initialization
        # A_B: ST2 but andomly pulling in step 789, from all arms [n]
        # PREDICTION: use predicted CTR directly, return arms with predicted CTR >= max(predicted CTR) - epsilon
        # DEBUG: LOLA on only one arm

        MODE = 'add' # add for 'additive', multi for 'multiplicative'
        if 'A_B' in METHOD:
            RANDOM_PULLING = True
            INIT_WITH_PRED = False
            COMPARE_PRED = False
            MAX_PULLS = int(METHOD.split('_')[-1])
            # STOP_CRITERIA = 'pulls'
            A_B_TEST = True
        elif METHOD == 'LOLA':
            INIT_WITH_PRED = True
            RANDOM_PULLING = False
            COMPARE_PRED = False
            MAX_PULLS = 300
            # STOP_CRITERIA = 'both'
            A_B_TEST = False
        elif METHOD == 'ST2':
            INIT_WITH_PRED = False
            RANDOM_PULLING = False
            COMPARE_PRED = False
            MAX_PULLS = 300
            # STOP_CRITEIA = 'both'
            A_B_TEST = False
        elif METHOD == 'PREDICTION':
            INIT_WITH_PRED = True
            RANDOM_PULLING = None
            COMPARE_PRED = True # use predicted CTR directly, return arms with predicted CTR >= max(predicted CTR) - epsilon
            MAX_PULLS = 300
            # STOP_CRITERIA = 'both'
            A_B_TEST = False
        elif METHOD == 'LLM':
            n_success = run_exp_LLM()
            success_rate = n_success/len(test_id_ls)
            print(success_rate)
            recording[METHOD] = {'success_rate': success_rate}

        if METHOD != 'LLM':
            # parameters to calibrate: c_Phi and initial_pulls

            if METHOD == 'DEBUG':
                test_id_ls = [1]
                VERBOSE = True
                INIT_WITH_PRED = True
            else:
                VERBOSE = False


            if METHOD != 'LOLA':
                initial_pulls_ls = [300]

            for initial_pulls_ in initial_pulls_ls:
                print('Initial Pulls: {}'.format(initial_pulls_))
                total_pulling_rec, n_success = run_one_exp(df, c_Phi, initial_pulls_, test_id_ls, INIT_WITH_PRED,  METHOD, MODE, MAX_PULLS, VERBOSE, RANDOM_PULLING, COMPARE_PRED, A_B_TEST)
            
                if METHOD == 'LOLA':
                    if METHOD not in recording:
                        recording[METHOD] = {}
                    recording[METHOD][initial_pulls_] = {'total_pulling_rec': total_pulling_rec, 'success_rate': n_success/len(test_id_ls)}
                else:
                    recording[METHOD] = {'total_pulling_rec': total_pulling_rec, 'success_rate': n_success/len(test_id_ls)}

    # # save the result using pkl
    # with open(os.path.join(result_dir, 'recording.pkl'), 'wb') as f:
    #     pickle.dump(recording, f)

    print('Save the result')
    plot(recording, max_min_CTR_gap_ls, os.path.join(result_dir, '{}_c_Phi{}'.format(filename, c_Phi)))

# run_all_exp(calibrate, c_Phi, initial_pulls_ls, 'calibrate')
run_all_exp(df, c_Phi, initial_pulls_ls, 'test')