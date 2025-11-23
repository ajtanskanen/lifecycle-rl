from .model import CNNModel
import torch
from torch import from_numpy
from torch.distributions import Normal
import numpy as np
from .kfac import KFAC
from common import explained_variance


class Brain:
    def __init__(self, **config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = CNNModel(self.config["state_shape"], self.config["n_actions"]).to(self.device)
        self.optimizer = KFAC(self.model,
                              lr=self.config["kfac_configs"]["lr"],
                              weight_decay=self.config["kfac_configs"]["weight_decay"],
                              damping=self.config["kfac_configs"]["damping"],
                              momentum=self.config["kfac_configs"]["momentum"],
                              eps=self.config["kfac_configs"]["epsilon"],
                              Ts=self.config["kfac_configs"]["Ts"],  # noqa
                              Tf=self.config["kfac_configs"]["Tf"],  # noqa
                              max_lr=self.config["kfac_configs"]["max_lr"],
                              trust_region=self.config["kfac_configs"]["trust_region"]
                              )
        self.mse_loss = torch.nn.MSELoss()

    def get_actions_and_values(self, state, batch=False):
        if not batch:
            state = np.expand_dims(state, 0)
        state = from_numpy(state).to(self.device)
        with torch.no_grad():
            dist, value = self.model(state)
            action = dist.sample()
        return action.cpu().numpy(), value.cpu().numpy().squeeze()

    def train(self, states, actions, rewards, dones, values, next_values):
        returns = self.get_returns(rewards, next_values, dones, n=self.config["n_workers"])
        values = np.hstack(values)
        advs = returns - values

        states = from_numpy(states).to(self.device)
        actions = from_numpy(actions).to(self.device)
        advs = from_numpy(advs).to(self.device)
        values_target = from_numpy(returns).to(self.device)

        dist, values_pred = self.model(states)
        ent = dist.entropy().mean()
        log_prob = dist.log_prob(actions)

        a_fisher_loss = -log_prob.mean()  # CELoss
        v_dist = Normal(values_pred.detach(), 1)
        v_samples = v_dist.sample()
        c_fisher_loss = -self.mse_loss(v_samples, values_pred)
        fisher_loss = a_fisher_loss + c_fisher_loss
        self.optimizer.zero_grad()
        self.optimizer.fisher_backprop = True
        fisher_loss.backward(retain_graph=True)
        self.optimizer.fisher_backprop = False

        a_loss = -(log_prob * advs).mean()
        c_loss = self.mse_loss(values_target, values_pred.squeeze(-1))
        total_loss = a_loss + self.config["critic_coeff"] * c_loss - self.config["ent_coeff"] * ent  # noqa
        self.optimize(total_loss)
        return a_loss.item(), c_loss.item(), ent.item(), explained_variance(values, returns)

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_returns(self, rewards: np.ndarray, next_values: np.ndarray, dones: np.ndarray, n: int) -> np.ndarray:
        if next_values.shape == ():
            next_values = next_values[None]

        returns = [[] for _ in range(n)]
        for worker in range(n):
            R = next_values[worker]  # noqa
            for step in reversed(range(len(rewards[worker]))):
                R = rewards[worker][step] + self.config["gamma"] * R * (1 - dones[worker][step])  # noqa
                returns[worker].insert(0, R)

        return np.hstack(returns).astype("float32")

    def set_from_checkpoint(self, checkpoint):
        self.model.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def prepare_to_play(self):
        self.model.eval()