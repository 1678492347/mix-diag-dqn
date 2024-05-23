import copy

import numpy as np
import torch
import torch.optim as optim
from qlearning.dqn import DQN
from agents.base_agent import BaseAgent
import config
from utils.replay_memory import ExperienceReplayMemory, MutExperienceReplayMemory, PrioritizedReplayMemory, MutPrioritizedReplayMemory


class AgentDQN(BaseAgent):
    def __init__(self, pretrained_xgb_cla_model, params):
        super().__init__()
        self.device = config.device
        self.pretrained_xgb_cla_model = pretrained_xgb_cla_model
        # parameters for DQN
        self.noisy = params.get('noisy', False)
        self.priority_replay = params.get('priority_replay', False)
        self.gamma = params.get('gamma', 0.9)
        self.nsteps = params.get('nsteps', 1)
        self.lr = params.get('lr', 0.01)
        self.batch_size = params.get('batch_size', 30)
        self.target_net_update_freq = params.get('target_net_update_freq', 1)
        self.experience_replay_size = params.get('experience_replay_size', 10000)
        self.sigma_init = params.get('sigma_init', 0.5)
        self.priority_beta_start = params.get('priority_beta_start', 0.4)
        self.priority_beta_frames = params.get('priority_beta_frames', 10000)
        self.priority_alpha = params.get('priority_alpha', 0.6)
        self.dqn_hidden_size = params.get('dqn_hidden_size', 128)
        self.fix_buffer = params.get('fix_buffer', True)

        self.origin_model = params.get('origin_model', 1)
        self.predict_mode = False
        self.warm_start = params.get('warm_start', 1)
        self.max_turn = config.MAX_DIAGNOSIS_NUM
        self.epsilon = params.get('epsilon', 0.3)

        self.phe_num_cardinality = 1
        self.phe_main_classes_cardinality = 23
        self.related_top10_score_cardinality = 10
        self.phelr_top10_prob_cardinality = 10
        self.phelr_top10_gini_cardinality = 1
        self.phelr_top10_entropy_cardinality = 1
        self.num_actions = config.ACTION_COUNT
        self.state_dimension = self.phe_num_cardinality + self.phe_main_classes_cardinality + self.related_top10_score_cardinality + self.phelr_top10_prob_cardinality + self.phelr_top10_gini_cardinality + self.phelr_top10_entropy_cardinality + self.num_actions*10 + (config.MAX_DIAGNOSIS_NUM + 1)
        self.feasible_actions = config.FEASIBLE_AGENT_ACTIONS

        self.declare_networks(params.get('trained_model_path', None))
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)

        self.model.to(self.device)
        self.target_model.to(self.device)

        self.model.train()
        self.target_model.eval()
        self.update_count = 0

        # declare memory
        if self.fix_buffer:
            self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size,
                                                                                                                                       self.priority_alpha,
                                                                                                                                       self.priority_beta_start,
                                                                                                                                       self.priority_beta_frames)
        else:
            self.memory = MutExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else MutPrioritizedReplayMemory(self.priority_alpha,
                                                                                                                                             self.priority_beta_start,
                                                                                                                                             self.priority_beta_frames)

    def declare_networks(self, path):
        self.model = DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions)
        self.target_model = DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions)
        if path is not None:
            if self.origin_model == 1:
                checkpoint = torch.load(path, map_location=self.device)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.predict_mode = True
                self.warm_start = 0
            else:
                self.load_specific_state_dict(path)
                self.predict_mode = True
                self.warm_start = 0

    def load_specific_state_dict(self, path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path, map_location=self.device)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def prepare_state_representation(self, state):
        env_action = state['env_action']
        env_inform_slots = env_action['inform_slots']

        phe_count_rep = torch.zeros(1, 1, device=self.device)
        phe_count_rep[0, 0] = env_inform_slots['phe_count']/20 if env_inform_slots['phe_count'] <= 20 else 1.0

        phe_main_classes_rep = torch.Tensor(np.array([env_inform_slots['phe_main_classes']])).to(self.device)
        phe_main_classes_rep /= env_inform_slots['phe_count']
        phe_main_classes_rep[phe_main_classes_rep >= 1] = 1.0

        related_top10_score_rep = torch.Tensor(np.array([env_inform_slots['related_top10_score']])).to(self.device)
        related_top10_score_rep /= env_inform_slots['phe_count']
        phelr_top10_prob_rep = torch.Tensor(np.array([env_inform_slots['phelr_top10_prob']])).to(self.device)

        phelr_top10_gini_entropy_rep = torch.Tensor(np.array([[env_inform_slots['phelr_top10_gini'], env_inform_slots['phelr_top10_entropy']]])).to(self.device)
        
        methods_count_rep = torch.zeros(3, 10, device=self.device)
        for idx, count in enumerate(env_inform_slots['methods_count']):
            methods_count_rep[idx, :count] = 1.0
        methods_count_rep = methods_count_rep.view(1, -1)

        ddx_list_pos_rep = torch.zeros(1, config.MAX_DIAGNOSIS_NUM + 1, device=self.device)
        ddx_list_pos_rep[0, env_inform_slots['ddx_list_pos']] = 1.0

        final_representation = torch.cat((
            phe_count_rep, phe_main_classes_rep, related_top10_score_rep, phelr_top10_prob_rep, phelr_top10_gini_entropy_rep, methods_count_rep, ddx_list_pos_rep
        ), 1)
        return final_representation

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_t_plus_1, episode_over):
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = a_t['inform_slots']['act_idx']
        reward_t = reward
        state_t_plus_1_rep = self.prepare_state_representation(s_t_plus_1)
        training_example = (state_t_rep, action_t, reward_t, state_t_plus_1_rep, episode_over)
        # only record experience of dqn train, and warm start
        if self.predict_mode:       # Predict Mode
            self.memory.push(training_example)
        else:                       # Train Mode
            if self.warm_start == 1:
                self.memory.push(training_example)

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over = zip(*transitions)
        batch_episode_over_tag = []
        for i in range(self.batch_size):
            if batch_episode_over[i]:
                batch_episode_over_tag.append(0)
            else:
                batch_episode_over_tag.append(1)

        batch_state = torch.cat(batch_state, dim=0).view(-1, self.state_dimension)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.cat(batch_next_state, dim=0).view(-1, self.state_dimension)
        batch_episode_over = torch.tensor(batch_episode_over_tag, device=self.device, dtype=torch.float).squeeze().view(-1, 1)  # flag for existence of next state

        return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights = batch_vars
        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)  # get the output values of the ground truth batch_action
        # target
        with torch.no_grad():
            max_next_q_values = self.target_model(batch_next_state).max(dim=1)[0].view(-1, 1)  # max q value
            expected_q_values = batch_reward + batch_episode_over * (self.gamma ** self.nsteps) * max_next_q_values
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = 0.5 * torch.pow(diff, 2).squeeze() * weights
        else:
            loss = 0.5 * torch.pow(diff, 2)
        loss = loss.mean()
        return loss

    def single_batch(self):
        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        cur_bellman_err = 0.0
        if self.fix_buffer:
            for i in range(int(len(self.memory) / self.batch_size)):
                cur_bellman_err += self.single_batch()
            # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
        else:
            for i in range(int(len(self.memory) / self.batch_size)):
                cur_bellman_err += self.single_batch()
            # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
        self.update_target_model()

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            # print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    def state_to_action(self, state):
        representation = self.prepare_state_representation(state)
        action_idx = self.run_policy(representation, state)
        return copy.deepcopy(self.feasible_actions[action_idx])

    def run_policy(self, rep, state):
        if self.warm_start != 1 and np.random.random() < self.epsilon:
            act_ind = np.random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.memory.buffer) >= self.experience_replay_size:
                    self.warm_start = 0
                act_ind = self.warm_start_policy(state)
            else:
                flag = torch.tensor(state['env_action']['inform_slots']['methods_count'])
                flag[flag < 10] = 1
                flag[flag == 10] = 0
                act_ind = self.model.predict(rep, flag)
        return act_ind

    def warm_start_policy(self, state):
        env_action = state['env_action']
        env_inform_slots = env_action['inform_slots']
        cur_turn = env_action['turn']
        X_test = np.column_stack((
            [env_inform_slots['phe_count']],
            [env_inform_slots['phe_main_classes']],
            [env_inform_slots['related_top10_score']],
            [env_inform_slots['phelr_top10_prob']],
            [env_inform_slots['phelr_top10_gini']],
            [env_inform_slots['phelr_top10_entropy']]
        ))
        prob = self.pretrained_xgb_cla_model.predict_proba(X_test)
        for i in range(len(prob[0])):
            top_n_idx = np.argsort(np.array([np.sort(j[i][:10] * (1 - j[i][10]))[::-1] for j in prob]).ravel())[::-1]
            top_n_idx = [divmod(j, 10) for j in top_n_idx]
            cur_method_idx = top_n_idx[cur_turn][0]
            return cur_method_idx
