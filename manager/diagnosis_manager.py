import config
from manager.state_tracker import StateTracker


class DiagnosisManager:
    """ A dialog manager to mediate the interaction between an agent and a sim_env """

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.state_tracker = StateTracker()

    def initialize_episode(self):
        """ Refresh state for new dialog """
        self.state_tracker.initialize_episode()
        self.state_tracker.update(env_action=self.env.initialize_episode())

    def next_turn(self, record_training_data=True):
        """ This function initiates each subsequent exchange between agent and env (agent first) """

        # CALL AGENT TO TAKE HER TURN
        state = self.state_tracker.get_state_for_agent()
        agent_action = self.agent.state_to_action(state)

        # Register AGENT action with the state_tracker
        self.state_tracker.update(agent_action=agent_action)

        # CALL ENV TO TAKE ITS TURN
        sys_action = self.state_tracker.dialog_history_dictionaries()[-1]
        env_action, episode_over, diagnosis_status, hit_target, close_to_target, target_pos = self.env.next(sys_action)

        # Reward
        if hit_target:
            reward = config.PHELR_GET_TARGET_REWARD if agent_action['inform_slots']['method_name'] == 'phelr' else config.NON_PHELR_GET_TARGET_REWARD
        elif close_to_target:
            reward = config.PHELR_CLOSER_TO_TARGET_REWARD if agent_action['inform_slots']['method_name'] == 'phelr' else config.NON_PHELR_CLOSER_TO_TARGET_REWARD
        else:
            reward = config.NO_CLOSER_TO_TARGET_REWARD

        # Update state tracker with latest env action
        self.state_tracker.update(env_action=env_action)

        # Inform agent of the outcome for this timestamp (s_t, a_t, r, s_{t+1}, episode_over)
        if record_training_data:
            self.agent.register_experience_replay_tuple(state, agent_action, reward,
                                                        self.state_tracker.get_state_for_agent(), episode_over)
        return episode_over, reward, diagnosis_status, target_pos
