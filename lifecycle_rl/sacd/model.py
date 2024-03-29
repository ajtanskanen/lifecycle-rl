import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()

        #self.net = nn.Sequential(
        #    nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
        #    nn.ReLU(),
        #    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #    nn.ReLU(),
        #   nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #   nn.ReLU(),
        #    Flatten(),
        #).apply(initialize_weights_he)

        self.net = nn.Sequential(
            nn.Linear(num_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ).apply(initialize_weights_he)

    def forward(self, states):
        return self.net(states)

class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(256, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(256, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)

class QNetworkMultiAct(BaseNetwork):

    def __init__(self, num_channels, actions, shared=False,
                 dueling_net=False):
        super().__init__()

        num_actions = np.sum(actions)

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)

class TwinnedQNetwork(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class TwinnedQNetwork_multidiscrete(BaseNetwork):
    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetworkMultiAct(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetworkMultiAct(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, temp=1.0):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states)/temp, dim=1)
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs

class MultidiscretePolicy(BaseNetwork):

    def __init__(self, num_channels, actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.num_actions = np.sum(actions)

        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_actions))

        self.shared = shared

    def act(self, states, cat=True, item=True):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)

        action1 = torch.argmax(action_logits[:,0:5], dim=1, keepdim=True)
        action2 = torch.argmax(action_logits[:,5:10], dim=1, keepdim=True)
        action3 = torch.argmax(action_logits[:,10:13], dim=1, keepdim=True)
        action4 = torch.argmax(action_logits[:,13:16], dim=1, keepdim=True)

        acts = torch.cat([action1, action2, action3, action4],dim=1)
        acts = acts.cpu().numpy().ravel()

        return acts

    def sample(self, states, cat=True, temp=1.0):
        def subsample(acts, temp=1.0):
            action_probs1 = F.softmax(acts/temp, dim=1)
            action_dist = Categorical(action_probs1)
            actions1 = action_dist.sample().view(-1,1)
            # Avoid numerical instability.
            z = (action_probs1 == 0.0).float() * 1e-8
            log_action_probs = torch.log(action_probs1 + z)

            return actions1,action_probs1,log_action_probs

        if not self.shared:
            states = self.conv(states)

        acts = self.head(states)
        acts1,action_probs1,log_action_probs1 = subsample(acts[:,0:5], temp=temp)
        acts2,action_probs2,log_action_probs2 = subsample(acts[:,5:10], temp=temp)
        acts3,action_probs3,log_action_probs3 = subsample(acts[:,10:13], temp=temp)
        acts4,action_probs4,log_action_probs4 = subsample(acts[:,13:16], temp=temp)
        acts = torch.cat([acts1,acts2,acts3,acts4],dim=1)
        acts = acts.cpu().numpy()

        if cat:
            log_action_probs = torch.cat([log_action_probs1, log_action_probs2, log_action_probs3, log_action_probs4],dim=1)
            action_probs = torch.cat([action_probs1, action_probs2, action_probs3, action_probs4],dim=1)
        else:
            log_action_probs = [log_action_probs1, log_action_probs2, log_action_probs3, log_action_probs4]
            action_probs = [action_probs1, action_probs2, action_probs3, action_probs4]

        return acts, action_probs, log_action_probs