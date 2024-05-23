import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Fix Buffer
class ExperienceReplayMemory:
    def __init__(self, capacity, success_rate_threshold=0.3):
        self.capacity = capacity
        self.buffer = []
        self.success_rate_threshold = success_rate_threshold
        self.best_success_rate = 0.0

    def push(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def clear(self):
        n = 5  # 保留大约1/n的元素
        indices_to_keep = random.sample(range(len(self.buffer)), len(self.buffer) // n)
        self.buffer = [self.buffer[i] for i in indices_to_keep]

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size), None, None

    def __len__(self):
        return len(self.buffer)


# Mutable Buffer
class MutExperienceReplayMemory:
    def __init__(self, success_rate_threshold=0.3):
        self.buffer = []
        self.success_rate_threshold = success_rate_threshold
        self.best_success_rate = 0.0

    def push(self, transition):
        self.buffer.append(transition)
        # if len(self.memory) > self.capacity:
        #     del self.memory[0]

    def clear(self):
        self.buffer = []

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size), None, None

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = self.priorities.max() if self.buffer else 1.0 ** self.prob_alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_prio

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):

        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        total = len(self.buffer)

        probs = prios / prios.sum()  # compute P(i)

        indices = np.random.choice(total, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total) ** (-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5) ** self.prob_alpha

    def __len__(self):
        return len(self.buffer)


class MutPrioritizedReplayMemory(object):
    def __init__(self, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.prob_alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = []
        self.frame = 1
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def beta_by_frame(self, frame_idx):
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, transition):
        max_prio = max(self.priorities) if self.buffer else 1.0 ** self.prob_alpha
        self.buffer.append(transition)
        self.pos += 1
        self.priorities.append(max_prio)
        # self.priorities[self.pos] = max_prio

    def clear(self):
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def sample(self, batch_size):

        total = len(self.buffer)
        prios = np.array(self.priorities)
        probs = prios / prios.sum()  # compute P(i)
        # indices = np.random.choice(total, batch_size, p=probs)
        indices = np.random.choice(total, batch_size)
        samples = [self.buffer[idx] for idx in indices]
        beta = self.beta_by_frame(self.frame)
        self.frame += 1

        # min of ALL probs, not just sampled probs
        prob_min = probs.min()
        max_weight = (prob_min * total) ** (-beta)

        weights = (total * probs[indices]) ** (-beta)
        weights /= max_weight
        weights = torch.tensor(weights, device=device, dtype=torch.float)

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = (prio + 1e-5) ** self.prob_alpha

    def __len__(self):
        return len(self.buffer)
