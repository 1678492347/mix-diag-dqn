import argparse
import collections
import copy
import json
import os
import pickle

import joblib
import numpy as np
import torch
from tensorboardX import SummaryWriter

import config
from agents.agent import AgentDQN
from manager.diagnosis_manager import DiagnosisManager
from sim_env.diagnosis_env import DiagnosisSimulator

writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument('--data_folder', dest='data_folder', type=str, default='dataset', help='folder to all data')
parser.add_argument('--episodes', dest='episodes', default=1000, type=int, help='Total number of episodes to run')
parser.add_argument('--agt', dest='agt', default=0, type=int, help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.6, help='Epsilon to determine stochasticity of epsilon-greedy agent policies')
parser.add_argument('--priority_replay', dest='priority_replay', default=0, type=int, help='')
parser.add_argument('--fix_buffer', dest='fix_buffer', default=1, type=int, help='')
parser.add_argument('--origin_model', dest='origin_model', default=0, type=int, help='0 for not mask')

# RL agent parameters
parser.add_argument('--experience_replay_size', dest='experience_replay_size', type=int, default=10000, help='the size for experience replay')
parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=512, help='the hidden size for DQN')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.03, help='lr for DQN')
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
goal_set = {'train': train_data, 'test': test_data}
legal_goal_idx = {
    'train': [i for i in range(len(goal_set['train']['target_orpha_code'])) if
              len(goal_set['train']['gpt_4_top10_orpha'][i]) >= 10 and len(goal_set['train']['related_top10_orpha'][i]) >= 10],
    'test': [i for i in range(len(goal_set['test']['target_orpha_code'])) if
             len(goal_set['test']['gpt_4_top10_orpha'][i]) >= 10 and len(goal_set['test']['related_top10_orpha'][i]) >= 10]
}
pretrained_xgb_cla_model = joblib.load('{}/multi_output_xgb_cla_model.joblib'.format(data_folder))

num_episodes = params['episodes']
fix_buffer = False
if params['fix_buffer'] == 1:
    fix_buffer = True
priority_replay = False
if params['priority_replay'] == 1:
    priority_replay = True

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


def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    target_pos_list = []
    res = {}
    warm_start_run_epochs = 0
    for episode in range(warm_start_epochs):
        diagnosis_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, diagnosis_status, target_pos = diagnosis_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                target_pos_list.append(target_pos)
                if diagnosis_status == config.SUCCESS_DIAGNOSIS:
                    successes += 1
        warm_start_run_epochs += 1
        if len(agent_dqn.memory) >= agent_dqn.experience_replay_size:
            break
    agent_dqn.warm_start = 0
    top10_accumulate = np.cumsum([collections.Counter(target_pos_list).get(i, 0) for i in range(1, 11)])
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['top_1_recall'] = top10_accumulate[0] / warm_start_run_epochs
    res['top_5_recall'] = top10_accumulate[4] / warm_start_run_epochs
    res['top_10_recall'] = top10_accumulate[9] / warm_start_run_epochs
    print("Warm start run %s epochs, success rate %.4f, ave reward %.4f, top_1 recall %.4f, top_5 recall %.4f, top_10 recall %.4f" % (
        warm_start_run_epochs, res['success_rate'], res['ave_reward'], res['top_1_recall'], res['top_5_recall'], res['top_10_recall']))
    print("Current experience replay buffer size %s" % (len(agent_dqn.memory)))


def dqn_simulation_epoch():
    """ simulate diagnosis and save experience """
    successes = 0
    cumulative_reward = 0
    target_pos_list = []
    res = {}
    for episode in range(simulation_epoch_size):
        diagnosis_manager.initialize_episode()
        episode_over = False
        while not episode_over:
            episode_over, r, diagnosis_status, target_pos = diagnosis_manager.next_turn()
            cumulative_reward += r
            if episode_over:
                target_pos_list.append(target_pos)
                if diagnosis_status == config.SUCCESS_DIAGNOSIS:
                    successes += 1
    top10_accumulate = np.cumsum([collections.Counter(target_pos_list).get(i, 0) for i in range(1, 11)])
    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['top_1_recall'] = top10_accumulate[0] / simulation_epoch_size
    res['top_5_recall'] = top10_accumulate[4] / simulation_epoch_size
    res['top_10_recall'] = top10_accumulate[9] / simulation_epoch_size
    # print("DQN %s simulation success rate %.4f, ave reward %.4f, top_1 recall %.4f, top_5 recall %.4f, top_10 recall %.4f"
    #       % (simulation_epoch_size, res['success_rate'], res['ave_reward'], res['top_1_recall'], res['top_5_recall'], res['top_10_recall']))
    return res


def test(test_size, data_split):
    successes = 0
    cumulative_reward = 0
    target_pos_list = []
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
                if diagnosis_status == config.SUCCESS_DIAGNOSIS:
                    successes += 1
    top10_accumulate = np.cumsum([collections.Counter(target_pos_list).get(i, 0) for i in range(1, 11)])
    res['success_rate'] = float(successes) / test_size
    res['ave_reward'] = float(cumulative_reward) / test_size
    res['top_1_recall'] = top10_accumulate[0] / test_size
    res['top_5_recall'] = top10_accumulate[4] / test_size
    res['top_10_recall'] = top10_accumulate[9] / test_size
    # print("Test %s_set success rate %.4f, ave reward %.4f, top_1 recall %.4f, top_5 recall %.4f, top_10 recall %.4f"
    #       % (data_split, res['success_rate'], res['ave_reward'], res['top_1_recall'], res['top_5_recall'], res['top_10_recall']))
    agent_dqn.epsilon = params['epsilon']
    return res


def training(count):
    # use rule policy, and record warm start experience
    if params['trained_model_path'] is None and warm_start == 1:
        print('warm_start starting...')
        env_sim.set_policy(data_split=params['train_set'], is_test=True)
        warm_start_simulation()
        print('warm_start finished, start RL training ...')
    start_episode = 0

    if params['trained_model_path'] is not None:
        trained_file = torch.load(params['trained_model_path'])
        if 'cur_epoch' in trained_file.keys():
            start_episode = trained_file['cur_epoch']

    # dqn simulation, train dqn, evaluation and save model
    for episode in range(start_episode, count):
        # print("Episode: %s" % episode)
        # simulation diagnosis
        env_sim.set_policy(data_split=params['train_set'], is_test=True)
        agent_dqn.predict_mode = True
        # simulate diagnosis and save experience
        dqn_simulation_epoch()

        # train by current experience pool
        agent_dqn.train()
        agent_dqn.predict_mode = False

        # eval
        test_env_sim.set_policy(data_split=params['test_set'], is_test=True)
        test_res = test(len(test_env_sim.candidate_goal_idx), data_split=params['test_set'])
        writer.add_scalar('test/accuracy', torch.tensor(test_res['success_rate'], device=config.device), episode)
        writer.add_scalar('test/ave_reward', torch.tensor(test_res['ave_reward'], device=config.device), episode)
        writer.add_scalar('test/top_1_recall', torch.tensor(test_res['top_1_recall'], device=config.device), episode)
        writer.add_scalar('test/top_5_recall', torch.tensor(test_res['top_5_recall'], device=config.device), episode)
        writer.add_scalar('test/top_10_recall', torch.tensor(test_res['top_10_recall'], device=config.device), episode)
        if test_res['top_1_recall'] > best_test_res['top_1_recall'] or test_res['top_5_recall'] > best_test_res['top_5_recall'] or test_res['top_10_recall'] > best_test_res['top_10_recall']:
            # clear buffer when accuracy promotes
            if test_res['top_1_recall'] >= 0.41 and test_res['top_10_recall'] >= 0.67:
                agent_dqn.memory.clear()
            best_test_model['model'] = agent_dqn.model.state_dict()
            best_test_res['success_rate'] = test_res['success_rate'] if test_res['success_rate'] > best_test_res['success_rate'] else best_test_res['success_rate']
            best_test_res['ave_reward'] = test_res['ave_reward'] if test_res['ave_reward'] > best_test_res['ave_reward'] else best_test_res['ave_reward']
            best_test_res['top_1_recall'] = test_res['top_1_recall'] if test_res['top_1_recall'] > best_test_res['top_1_recall'] else best_test_res['top_1_recall']
            best_test_res['top_5_recall'] = test_res['top_5_recall'] if test_res['top_5_recall'] > best_test_res['top_5_recall'] else best_test_res['top_5_recall']
            best_test_res['top_10_recall'] = test_res['top_10_recall'] if test_res['top_10_recall'] > best_test_res['top_10_recall'] else best_test_res['top_10_recall']
            best_test_res['epoch'] = episode
            save_model(params['write_model_dir'], agent_dqn, cur_epoch=episode, best_epoch=best_test_res['epoch'], best_top_1_recall=test_res['top_1_recall'],
                       best_top_5_recall=test_res['top_5_recall'], best_top_10_recall=test_res['top_10_recall'], phase="test")
        elif test_res['top_1_recall'] > 0.42 and test_res['top_5_recall'] > 0.60 and test_res['top_10_recall'] > 0.70:
            save_model(params['write_model_dir'], agent_dqn, cur_epoch=episode, best_epoch=episode, best_top_1_recall=test_res['top_1_recall'],
                       best_top_5_recall=test_res['top_5_recall'], best_top_10_recall=test_res['top_10_recall'], phase="test")
        # save checkpoint each episode
        # save_model(params['write_model_dir'], agent_dqn, cur_epoch=episode, test_success_rate=test_res['success_rate'], is_checkpoint=True)
        # print()
        print("\rFinished episode: %s/%s. Best top_1_recall: %.3f, top_5_recall: %.3f, top_10_recall: %.3f" % (
            episode+1, count, best_test_res['top_1_recall'], best_test_res['top_5_recall'], best_test_res['top_10_recall']), end='')


def save_model(path, agent, cur_epoch, best_epoch=0, best_top_1_recall=0.0, best_top_5_recall=0.0, best_top_10_recall=0.0,
               eval_success_rate=0.0, test_success_rate=0.0, phase="", is_checkpoint=False):
    """ Save model """
    if not os.path.exists(path):
        os.makedirs(path)
    checkpoint = {'cur_epoch': cur_epoch, 'state_dict': agent.model.state_dict()}
    if is_checkpoint:
        file_name = 'checkpoint.pth.tar'
        checkpoint['eval_success_rate'] = eval_success_rate
        checkpoint['test_success_rate'] = test_success_rate
    else:
        file_name = 'agt_%s_%s_%s_%.3f_%.3f_%.3f.pth.tar' % (agt, phase, best_epoch, best_top_1_recall, best_top_5_recall, best_top_10_recall)
        checkpoint['best_success_rate'] = best_top_1_recall
        checkpoint['best_epoch'] = best_epoch
    file_path = os.path.join(path, file_name)
    torch.save(checkpoint, file_path)


agt = params['agt']
training(num_episodes)
print()
