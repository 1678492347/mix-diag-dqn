import argparse
import collections
import copy
import json
import os
import pickle
import re

import joblib
import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

import config
from agents.agent import AgentDQN
from manager.diagnosis_manager import DiagnosisManager
from sim_env.diagnosis_env import DiagnosisSimulator

from utils.utils import all_orpha

writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', dest='data_folder', type=str, default='dataset', help='folder to all data')
parser.add_argument('--episodes', dest='episodes', default=1000, type=int, help='Total number of episodes to run (default=10)')
parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.5, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')
parser.add_argument('--priority_replay', dest='priority_replay', default=0, type=int, help='')
parser.add_argument('--fix_buffer', dest='fix_buffer', default=1, type=int, help='')
parser.add_argument('--origin_model', dest='origin_model', default=0, type=int, help='0 for not mask')

# RL agent parameters
parser.add_argument('--experience_replay_size', dest='experience_replay_size', type=int, default=10000, help='the size for experience replay')
parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=256, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='lr for DQN')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False, help='predict model for DQN')
parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=100, help='the size of validation set')
parser.add_argument('--target_net_update_freq', dest='target_net_update_freq', type=int, default=1, help='update frequency')
parser.add_argument('--warm_start', dest='warm_start', type=int, default=1, help='0: no warm start; 1: warm start for training')
parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=1000, help='the number of epochs for warm start')
parser.add_argument('--supervise', dest='supervise', type=int, default=1, help='0: no supervise; 1: supervise for training')
parser.add_argument('--supervise_epochs', dest='supervise_epochs', type=int, default=100, help='the number of epochs for supervise')

parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None, help='the path for trained model')
parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str, default='./checkpoints/', help='write model to disk')
parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=10, help='number of epochs for saving model')
parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.6, help='the threshold for train set success rate')
parser.add_argument('--train_set', dest='train_set', default='train', type=str, help='train/test')
parser.add_argument('--test_set', dest='test_set', default='test', type=str, help='train/test')

args = parser.parse_args()
params = vars(args)

print('Dialog Parameters: ')
print(json.dumps(params, indent=2))


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# load train and test data set
data_folder = params['data_folder']
train_data = load_pickle('{}/train.pickle'.format(data_folder))
test_data = load_pickle('{}/test.pickle'.format(data_folder))

for i in range(len(test_data['gpt_4_top10_orpha'])):
    gpt_4_top10_orpha = test_data['gpt_4_top10_orpha'][i]
    gpt_4_top10_orpha = [i for i in gpt_4_top10_orpha if i in all_orpha]
    if len(gpt_4_top10_orpha) < 10:
        gpt_4_top10_orpha += [config.ORPHA_PLACEHOLDER] * (10 - len(gpt_4_top10_orpha))
    test_data['gpt_4_top10_orpha'][i] = gpt_4_top10_orpha

    top_10_match_score = list(test_data['related_top10_score'][i])
    related_top10_orpha = test_data['related_top10_orpha'][i]
    if len(top_10_match_score) < 10:
        top_10_match_score += [0] * (10 - len(top_10_match_score))
    if len(related_top10_orpha) < 10:
        related_top10_orpha += [config.ORPHA_PLACEHOLDER] * (10 - len(related_top10_orpha))
    test_data['related_top10_score'][i] = top_10_match_score
    test_data['related_top10_orpha'][i] = related_top10_orpha

goal_set = {'train': train_data, 'test': test_data}
legal_goal_idx = {
    'train': [i for i in range(len(goal_set['train']['target_orpha_code'])) if
              len(goal_set['train']['gpt_4_top10_orpha'][i]) >= 10 and len(goal_set['train']['gpt_match_top10_orpha'][i]) >= 10
              and len(goal_set['train']['related_top10_orpha'][i]) >= 10
              ],
    'test': [i for i in range(len(goal_set['test']['target_orpha_code'])) if
             len(goal_set['test']['gpt_4_top10_orpha'][i]) >= 10 and len(goal_set['test']['gpt_match_top10_orpha'][i]) >= 10
             and len(goal_set['test']['related_top10_orpha'][i]) >= 10
             ]
}
pretrained_xgb_cla_model = joblib.load('{}/multi_output_xgb_cla_model.joblib'.format(data_folder))

num_episodes = params['episodes']
fix_buffer = False
if params['fix_buffer'] == 1:
    fix_buffer = True
priority_replay = False
if params['priority_replay'] == 1:
    priority_replay = True
    
params['dqn_hidden_size'] = 512
params['trained_model_path'] = './checkpoints1/agt_9_test_306_0.418_0.637_0.692.pth.tar'

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['epsilon'] = params['epsilon']
agent_params['experience_replay_size'] = params['experience_replay_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['lr'] = params['lr']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['supervise'] = params['supervise']
agent_params['fix_buffer'] = fix_buffer
agent_params['priority_replay'] = priority_replay
agent_params['target_net_update_freq'] = params['target_net_update_freq']
agent_params['origin_model'] = params['origin_model']
agent_dqn = AgentDQN(pretrained_xgb_cla_model, agent_params)
################################################################################
#   Parameters for User Simulators
################################################################################
env_sim = DiagnosisSimulator(goal_set, legal_goal_idx=legal_goal_idx, data_split=params['train_set'], is_test=False)
test_env_sim = DiagnosisSimulator(goal_set, legal_goal_idx=legal_goal_idx, data_split=params['test_set'], is_test=True)
################################################################################
# Dialog Manager
################################################################################
dm_params = {}
diagnosis_manager = DiagnosisManager(agent_dqn, env_sim)
test_diagnosis_manager = DiagnosisManager(agent_dqn, test_env_sim)
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}
################################################################################
# Exp Design
################################################################################
simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']
supervise = params['supervise']
supervise_epochs = params['supervise_epochs']
success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']

""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'top_1_recall': float('-inf'), 'top_5_recall': float('-inf'), 'top_10_recall': float('-inf'), 'epoch': 0}
best_model['model'] = agent_dqn.model.state_dict()

best_test_model = {}
best_test_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'top_1_recall': float('-inf'), 'top_5_recall': float('-inf'), 'top_10_recall': float('-inf'), 'epoch': 0}
best_test_model['model'] = agent_dqn.model.state_dict()
performance_records = {'success_rate': {}, 'ave_turns': {}, 'ave_reward': {}}

output = False

episode_reward = 0


def test(test_size):
    successes = 0
    cumulative_reward = 0
    res = {}
    agent_dqn.epsilon = 0
    for episode in range(test_size):
        test_diagnosis_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, diagnosis_status, target_pos = test_diagnosis_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                target_pos_list.append(target_pos)
                act_idx_list.append(test_diagnosis_manager.env.state['cur_ddx_act_idx'])
                target_orpha = test_diagnosis_manager.env.goal['target_orpha_code']
                if target_orpha in disorder_is_a:
                    ddx_list = test_diagnosis_manager.env.state['cur_ddx_slots']
                    broader_pos = min([ddx_list.index(broader_disorder) + 1 if broader_disorder in ddx_list else 11 for broader_disorder in disorder_is_a[target_orpha]])
                    broader_target_pos_list.append(min(target_pos, broader_pos))
                    if broader_pos < target_pos and test_diagnosis_manager.env.state['cur_ddx_act_idx'][broader_pos-1] != 3:
                        print()
                else:
                    broader_target_pos_list.append(target_pos)
                # test_diagnosis_manager.env.goal
                if diagnosis_status == config.SUCCESS_DIAGNOSIS:
                    successes += 1
    top10_accumulate = np.cumsum([collections.Counter(target_pos_list).get(i, 0) for i in range(1, 11)])
    broader_top10_accumulate = np.cumsum([collections.Counter(broader_target_pos_list).get(i, 0) for i in range(1, 11)])
    res['success_rate'] = float(successes) / test_size
    res['ave_reward'] = float(cumulative_reward) / test_size
    res['top_1_recall'] = top10_accumulate[0] / test_size
    res['top_5_recall'] = top10_accumulate[4] / test_size
    res['top_10_recall'] = top10_accumulate[9] / test_size
    res['broader_top_1_recall'] = broader_top10_accumulate[0] / test_size
    res['broader_top_5_recall'] = broader_top10_accumulate[4] / test_size
    res['broader_top_10_recall'] = broader_top10_accumulate[9] / test_size
    # print("Test %s_set success rate %.4f, ave reward %.4f, top_1 recall %.4f, top_5 recall %.4f, top_10 recall %.4f"
    #       % (data_split, res['success_rate'], res['ave_reward'], res['top_1_recall'], res['top_5_recall'], res['top_10_recall']))
    agent_dqn.epsilon = params['epsilon']
    return res


with open('dataset/disorder_is_a.pkl', 'rb') as f:
    disorder_is_a = pickle.load(f)
act_idx_list = []
target_pos_list = []
broader_target_pos_list = []
test_env_sim.set_policy(data_split=params['test_set'], is_test=True)
test_res = test(len(test_env_sim.candidate_goal_idx))
print(test_res)
plt.figure(figsize=(50, 5))
colors = ['#91caff', '#ffd591', '#ffa39e']
plt.imshow(np.array(act_idx_list).T, cmap=plt.cm.colors.ListedColormap(colors), aspect='auto')
plt.ylabel('ranks')
plt.title('Heatmap of Value Distribution')
plt.show()
print()