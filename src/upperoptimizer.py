# -*- coding: utf-8 -*-
# **********************************
# Author: Lizi
# Contact:  alisonbrielee@gmail.com
# **********************************


import torch
import numpy as np
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class HyperGradient(object):
    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.upper_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, target, source, eta, network_optimizer):
        loss, _, _ = self.model._lower_loss(target, source)
        theta = _concat(self.model.lower_parameters()).data  #
        try:  # optimizer.step()
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.lower_parameters()).mul_(
                self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(
            torch.autograd.grad(loss, self.model.lower_parameters())).data + self.network_weight_decay * theta
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        return unrolled_model

    def step(self, target, source, target_valid, source_valid, target_mask_valid, source_mask_valid,
             eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(target, source, target_valid, source_valid, target_mask_valid,
                                         source_mask_valid, eta, network_optimizer)
        else:
            self._backward_step(target_valid, source_valid, target_mask_valid, source_mask_valid)
        self.optimizer.step()

    def _backward_step(self, target_valid, source_valid, target_mask_valid, source_mask_valid):
        loss = self.model._upper_loss(target_valid, source_valid, target_mask_valid, source_mask_valid)
        loss.backward()


    def _backward_step_unrolled(self, target, source, target_valid, source_valid, target_mask_valid, source_mask_valid,
                                eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(target, source, eta, network_optimizer)  # w'
        unrolled_loss = unrolled_model._upper_loss(target_valid, source_valid, target_mask_valid, source_mask_valid)  # L_val(w')

        unrolled_loss.backward()  #

        dalpha = [torch.zeros_like(v) for v in unrolled_model.upper_parameters()]  # dalpha{L_val(w', alpha)}
        vector = [v.grad.data for v in unrolled_model.lower_parameters()]  # dw'{L_val(w', alpha)}

        implicit_grads = self._hessian_vector_product(vector, target, source)
        # gradient = dalpha - lr * hessian
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.model.upper_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.model.new()
        params, offset = {}, 0
        for v in model_new.lower_parameters():
            v_length = np.prod(v.size())
            v.data = theta[offset: offset + v_length].view(v.size())
            offset += v_length
        return model_new.cuda()

    def _hessian_vector_product(self, vector, target, source, r=1e-2):
        R = r / _concat(vector).norm()
        # w+ = w + eps*dw'
        for p, v in zip(self.model.lower_parameters(), vector):
            p.data.add_(R, v)
        loss, _, _ = self.model._lower_loss(target, source)
        grads_p = torch.autograd.grad(loss, self.model.upper_parameters(), allow_unused=True)
        # w- = w - eps*dw'
        for p, v in zip(self.model.lower_parameters(), vector):
            p.data.sub_(2 * R, v)

        loss, _, _ = self.model._lower_loss(target, source)
        grads_n = torch.autograd.grad(loss, self.model.upper_parameters(), allow_unused=True)

        for p, v in zip(self.model.lower_parameters(), vector):
            p.data.add_(R, v)

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
