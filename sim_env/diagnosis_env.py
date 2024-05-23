import random
import uuid

import config


class DiagnosisSimulator:
    """ A rule-based Diagnosis simulator for testing mix diagnosis policy """

    def __init__(self, goal_set, legal_goal_idx, data_split, is_test):
        """ Constructor shared by all env simulators """
        self.goal_set = goal_set
        self.legal_goal_idx = legal_goal_idx
        self.data_split = data_split
        self.is_test = is_test
        self.candidate_goal_idx = legal_goal_idx[data_split][:]     # Optional candidate goal index

        self.state = {}
        self.goal = {}
        self.ddx_list_capacity = config.MAX_DIAGNOSIS_NUM

    def set_policy(self, data_split, is_test):
        self.data_split = data_split
        self.is_test = is_test
        self.candidate_goal_idx = self.legal_goal_idx[self.data_split][:]

    def initialize_episode(self):
        """ Initialize a new episode
        state['cur_ddx_slots']: keep the current differential diagnosis list
        state['rest_slots']: keep all the slots (which is still in the stack yet)
        """
        self._sample_goal()

        self.state.clear()
        self.state['cur_ddx_slots'] = []
        self.state['cur_ddx_act_idx'] = []
        # self.state['rest_slots'] = [self.goal['gpt_top10_orpha'][:], self.goal['gpt_match_top10_orpha'][:], self.goal['related_top10_orpha'][:], self.goal['phelr_top10_orpha'][:]]
        self.state['rest_slots'] = [self.goal['gpt_4_top10_orpha'][:], self.goal['related_top10_orpha'][:], self.goal['phelr_top10_orpha'][:]]
        self.state['inform_slots'] = {
            'ddx_list_pos': len(self.state['cur_ddx_slots']),
            'methods_count': [0, 0, 0],
            'phe_count': self.goal['phe_count'],
            'phe_main_classes': self.goal['phe_main_classes'],
            'related_top10_score': self.goal['related_top10_score'],
            'phelr_top10_prob': self.goal['phelr_top10_prob'],
            'phelr_top10_gini': self.goal['phelr_top10_gini'],
            'phelr_top10_entropy': self.goal['phelr_top10_entropy']
        }
        self.state['turn'] = 0

        return {
            'inform_slots': self.state['inform_slots'],
            'turn': self.state['turn']
        }

    def _sample_goal(self):
        """ sample a env goal  """
        self.goal.clear()
        goal_idx = random.choice(self.candidate_goal_idx)
        self.goal['target_orpha_code'] = self.goal_set[self.data_split]['target_orpha_code'][goal_idx]
        self.goal['phe_count'] = self.goal_set[self.data_split]['phe_count'][goal_idx]
        self.goal['phe_main_classes'] = self.goal_set[self.data_split]['phe_main_classes'][goal_idx]
        self.goal['related_top10_score'] = self.goal_set[self.data_split]['related_top10_score'][goal_idx]
        self.goal['phelr_top10_prob'] = self.goal_set[self.data_split]['phelr_top10_prob'][goal_idx]
        self.goal['phelr_top10_gini'] = self.goal_set[self.data_split]['phelr_top10_gini'][goal_idx]
        self.goal['phelr_top10_entropy'] = self.goal_set[self.data_split]['phelr_top10_entropy'][goal_idx]
        self.goal['gpt_4_top10_orpha'] = self.goal_set[self.data_split]['gpt_4_top10_orpha'][goal_idx]
        self.goal['related_top10_orpha'] = self.goal_set[self.data_split]['related_top10_orpha'][goal_idx]
        self.goal['phelr_top10_orpha'] = self.goal_set[self.data_split]['phelr_top10_orpha'][goal_idx]

        if self.is_test:
            self.candidate_goal_idx.remove(goal_idx)

    def next(self, agent_action):
        """ Generate next Env Action based on last Agent Action """
        hit_target = False
        close_to_target = False
        diagnosis_status = config.UNFINISHED_YET
        episode_over = False
        target_pos = self.ddx_list_capacity + 1

        act_idx = agent_action['inform_slots']['act_idx']

        if len(self.state['rest_slots'][act_idx]) > 0:
            cur_orpha = self.state['rest_slots'][act_idx][0]
        else:
            cur_orpha = str(uuid.uuid4())
        min_target_pos_before = min(self.find_target_pos())

        if len(self.state['rest_slots'][act_idx]) > 0:
            self.state['rest_slots'][act_idx] = self.state['rest_slots'][act_idx][1:]
        min_target_pos_after = min(self.find_target_pos())

        if cur_orpha == self.goal['target_orpha_code']:
            hit_target = True
        if min_target_pos_after < min_target_pos_before:
            close_to_target = True
        if cur_orpha not in self.state['cur_ddx_slots'] and cur_orpha != config.ORPHA_PLACEHOLDER:
            self.state['cur_ddx_slots'].append(cur_orpha)
            self.state['cur_ddx_act_idx'].append(act_idx)
        if len(self.state['cur_ddx_slots']) >= self.ddx_list_capacity:
            episode_over = True
            if self.goal['target_orpha_code'] in self.state['cur_ddx_slots']:
                target_pos = self.state['cur_ddx_slots'].index(self.goal['target_orpha_code']) + 1
                diagnosis_status = config.SUCCESS_DIAGNOSIS
            else:
                diagnosis_status = config.FAILED_DIAGNOSIS

        self.state['inform_slots']['ddx_list_pos'] = len(self.state['cur_ddx_slots'])
        self.state['inform_slots']['methods_count'][act_idx] += 1

        next_action = {
            'inform_slots': self.state['inform_slots'],
            'turn': self.state['turn']
        }

        return next_action, episode_over, diagnosis_status, hit_target, close_to_target, target_pos

    def find_target_pos(self):
        pos_list = []
        for ls in self.state['rest_slots']:
            try:
                idx = ls.index(self.goal['target_orpha_code'])
            except ValueError:
                idx = 10
            pos_list.append(idx)
        return pos_list
