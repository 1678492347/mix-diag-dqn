import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# Diagnosis status
################################################################################
FAILED_DIAGNOSIS = -1
SUCCESS_DIAGNOSIS = 1
UNFINISHED_YET = 0

################################################################################
# Rewards
################################################################################
PHELR_GET_TARGET_REWARD = 2
PHELR_CLOSER_TO_TARGET_REWARD = 1
NON_PHELR_GET_TARGET_REWARD = 2
NON_PHELR_CLOSER_TO_TARGET_REWARD = 1
NO_CLOSER_TO_TARGET_REWARD = -0.5


################################################################################
# Diagnosis config
################################################################################
MAX_TURN = 10
MAX_DIAGNOSIS_NUM = 10

ACTION_COUNT = 3
GPT4_ACTION_INDEX = 0
PHENOTYPE_MATCH_ACTION_INDEX = 1
PHELR_ACTION_INDEX = 2
FEASIBLE_AGENT_ACTIONS = [
    {'inform_slots': {'method_name': 'gpt4', 'act_idx': GPT4_ACTION_INDEX}},
    {'inform_slots': {'method_name': 'phenotype_match', 'act_idx': PHENOTYPE_MATCH_ACTION_INDEX}},
    {'inform_slots': {'method_name': 'phelr', 'act_idx': PHELR_ACTION_INDEX}}
]

ORPHA_PLACEHOLDER = 'ORPHA_PLACEHOLDER'
