"""
Created on May 20, 2016

state tracker

@author: xiul, t-zalipt
"""

import numpy as np
import copy


class StateTracker:
    """ The state tracker maintains a record of which request slots are filled and which inform slots are filled """

    def __init__(self):
        """ constructor for state tracker takes movie knowledge base and initializes a new episode

        Arguments:

        Class Variables:
        history_vectors         --  A record of the current dialog so far in vector format (act-slot, but no values)
        history_dictionaries    --  A record of the current dialog in dictionary format
        current_slots           --  A dictionary that keeps a running record of which slots are filled current_slots['inform_slots'] and which are requested current_slots['request_slots'] (but not filed)
        turn_count              --  A running count of which turn we are at in the present dialog
        """
        self.history_dictionaries = []
        self.turn_count = 0

    def initialize_episode(self):
        """ Initialize a new episode (diagnosis), flush the current state and tracked slots """
        self.history_dictionaries = []
        self.turn_count = 0

    def dialog_history_dictionaries(self):
        """  Return the dictionary representation of the dialog history (includes values) """
        return self.history_dictionaries

    def get_state_for_agent(self):
        """ Get the state representatons to send to agent """
        state = {
            'env_action': self.history_dictionaries[-1],
            'turn': self.turn_count,
            'agent_action': self.history_dictionaries[-2] if len(self.history_dictionaries) > 1 else None
        }
        return copy.deepcopy(state)

    def update(self, agent_action=None, env_action=None):
        """ Update the state based on the latest action """

        # Make sure that the function was called properly
        assert (not (env_action and agent_action))
        assert (env_action or agent_action)

        # Update state to reflect a new action by the agent.
        if agent_action:
            # Handles the act_slot response (with values needing to be filled)
            agent_action = copy.deepcopy(agent_action)
            agent_action_values = {'turn': self.turn_count, 'speaker': "agent", 'inform_slots': agent_action['inform_slots']}
            self.history_dictionaries.append(agent_action_values)
            self.turn_count += 1

        #   Update the state to reflect a new action by the env
        elif env_action:
            env_action = copy.deepcopy(env_action)
            env_action_values = {'turn': self.turn_count, 'speaker': "env", 'inform_slots': env_action['inform_slots']}
            self.history_dictionaries.append(copy.deepcopy(env_action_values))
        else:
            pass
