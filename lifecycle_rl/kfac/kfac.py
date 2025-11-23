# K-FAC Optimizer as a pre-conditioner for SGD
import torch
import torch.nn.functional as F  # noqa
from torch.nn import Linear, Conv2d
from torch import optim
import math


class KFAC(optim.Optimizer):
    def __init__(self,
                 model: torch.nn.Module,
                 lr=0.25,
                 weight_decay=0,
                 damping=1e-2,
                 momentum=0.9,
                 eps=0.95,
                 Ts=1,  # noqa
                 Tf=10,  # noqa
                 max_lr=1,
                 trust_region=0.001
                 ):

        super(KFAC, self).__init__(model.parameters(), {})
        self.acceptable_layer_types = [Linear, Conv2d]
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.damping = damping
        self.eps = eps
        self.Ts = Ts
        self.Tf = Tf
        self.max_lr = max_lr
        self.trust_region = trust_region
        self._k = 0
        self._aa_hat, self._gg_hat = {}, {}
        self._eig_a, self._Q_a = {}, {}
        self._eig_g, self._Q_g = {}, {}
        self._trainable_layers = []
        self.optim = optim.SGD(self.model.parameters(), lr=lr * (1 - momentum), momentum=momentum)
        self.fisher_backprop = False
        self._keep_track_aa_gg()

    def _keep_track_aa_gg(self):
        for m in self.model.modules():
            if type(m) in self.acceptable_layer_types:
                m.register_forward_pre_hook(self._save_aa)
                m.register_backward_hook(self._save_gg)
                self._trainable_layers.append(m)

    def _save_aa(self, layer, layer_input):
        if torch.is_grad_enabled() and self._k % self.Ts == 0:
            a = layer_input[0].data
            batch_size = a.size(0)
            if isinstance(layer, Conv2d):
                a = img2col(a, layer.kernel_size, layer.stride, layer.padding)

            if layer.bias is not None:
                a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)

            aa = (a.t() @ a) / batch_size

            if self._k == 0:
                self._aa_hat[layer] = aa.clone()

            polyak_avg(aa, self._aa_hat[layer], self.eps)

    def _save_gg(self, layer, delta, grad_backprop):  # noqa
        if self.fisher_backprop:
            ds = grad_backprop[0]
            batch_size = ds.size(0)
            if self._k % self.Ts == 0:
                if isinstance(layer, Conv2d):
                    ow, oh = ds.shape[-2:]
                    ds = ds.transpose_(1, 2).transpose_(2, 3).contiguous()
                    ds = ds.view(-1, ds.size(-1))

                ds *= batch_size
                gg = (ds.t() @ ds) / batch_size / oh / ow if isinstance(layer, Conv2d) else (ds.t() @ ds) / batch_size

                if self._k == 0:
                    self._gg_hat[layer] = gg.clone()

                polyak_avg(gg, self._gg_hat[layer], self.eps)

    def _update_inverses(self, l):
        self._eig_a[l], self._Q_a[l] = torch.linalg.eigh(self._aa_hat[l], UPLO='U')
        self._eig_g[l], self._Q_g[l] = torch.linalg.eigh(self._gg_hat[l], UPLO='U')
        self._eig_a[l] *= (self._eig_a[l] > 1e-6).float()
        self._eig_g[l] *= (self._eig_g[l] > 1e-6).float()

    def step(self, closure=None):
        if self.weight_decay > 0:
            for p in self.model.parameters():
                p.grad.data.add_(p.data, self.weight_decay)

        updates = {}
        for layer in self._trainable_layers:
            if self._k % self.Tf == 0:
                self._update_inverses(layer)

            grad = layer.weight.grad.data
            if isinstance(layer, Conv2d):
                grad = grad.view(grad.size(0), -1)

            if layer.bias is not None:
                grad = torch.cat([grad, layer.bias.grad.data.view(-1, 1)], 1)

            V1 = self._Q_g[layer].t() @ grad @ self._Q_a[layer]  # noqa
            V2 = V1 / (self._eig_g[layer].unsqueeze(-1) @ self._eig_a[layer].unsqueeze(0) + (  # noqa
                    self.damping + self.weight_decay)
                       )
            delta_h_hat = self._Q_g[layer] @ V2 @ self._Q_a[layer].t()

            if layer.bias is not None:
                delta_h_hat = [delta_h_hat[:, :-1], delta_h_hat[:, -1:]]
                delta_h_hat[0] = delta_h_hat[0].view(layer.weight.grad.data.size())
                delta_h_hat[1] = delta_h_hat[1].view(layer.bias.grad.data.size())
            else:
                delta_h_hat = [delta_h_hat.view(layer.weight.grad.data.size())]

            updates[layer] = delta_h_hat

        second_taylor_expan_term = 0
        for layer in self._trainable_layers:
            v = updates[layer]
            second_taylor_expan_term += (v[0] * layer.weight.grad.data * self.lr * self.lr).sum()
            if layer.bias is not None:
                second_taylor_expan_term += (v[1] * layer.bias.grad.data * self.lr * self.lr).sum()

        nu = min(self.max_lr, math.sqrt(2 * self.trust_region / (second_taylor_expan_term + 1e-6)))

        for layer in self._trainable_layers:
            v = updates[layer][0]
            layer.weight.grad.data.copy_(v)
            layer.weight.grad.data.mul_(nu)
            if layer.bias is not None:
                v = updates[layer][1]
                layer.bias.grad.data.copy_(v)
                layer.bias.grad.data.mul_(nu)

        self.optim.step()
        self._k += 1

    def state_dict(self) -> dict:
        return dict(sgd_state_dict=self.optim.state_dict(),
                    k=self._k,
                    aa_hat=self._aa_hat,
                    gg_hat=self._gg_hat,
                    eig_a=self._eig_a,
                    eig_g=self._eig_g,
                    Q_a=self._Q_a,
                    Q_g=self._Q_g
                    )

    def load_state_dict(self, state_dict: dict) -> None:
        self.optim.load_state_dict(state_dict["sgd_state_dict"])
        self._k = state_dict["k"]
        aa_hat = state_dict["aa_hat"].values()
        # gard values are stored in the backward pass so last values correspond to first layers
        gg_hat = reversed(state_dict["gg_hat"].values())
        eig_a = state_dict["eig_a"].values()
        eig_g = state_dict["eig_g"].values()
        Q_a = state_dict["Q_a"].values()
        Q_g = state_dict["Q_g"].values()

        for a, g, ea, eg, q_a, q_g, l in zip(aa_hat, gg_hat, eig_a, eig_g, Q_a, Q_g, self._trainable_layers):
            self._aa_hat[l] = a
            self._gg_hat[l] = g
            self._eig_a[l] = ea
            self._eig_g[l] = eg
            self._Q_a[l] = q_a
            self._Q_g[l] = q_g


def img2col(tensor,
            kernel_size: tuple,
            stride: tuple,
            padding: tuple
            ):
    x = tensor.data
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1],
                      padding[0], padding[0]
                      )
                  )
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(-1, x.size(3) * x.size(4) * x.size(5))
    return x


def polyak_avg(new, old, tau):  # noqa
    old *= tau / (1 - tau)
    old += new
    old *= (1 - tau)