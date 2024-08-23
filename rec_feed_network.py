import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from collections import OrderedDict

class RFNetwork(nn.Module):
    def __init__(self, net_params, rec_params):

      super(RFNetwork, self).__init__()
      self.init_network(net_params)
      self.init_recordings(rec_params)

    def forward(self, inputs, test=False):
      
        for timestep in range(input.shape[0]):

            self.in_ = input[timestep]
            if self.pattern_complete:
                self.in_ = self.pattern_complete_in(self.in_)
            
            if self.forward_input:
                self.out =  F.linear(self.in_, self.out_in)
            else:
                self.out = self.activation_out(radnom=True)


            self.hebbian_in_in()
            self.homeostasis_in_in()
            self.hebbian_out_in()
            self.homeostasis_out_in()

        self.record()


    def activation_in(self, x, random=False):
      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, int(self.in_size*self.in_sparsity)).indices] = 1
      return x_prime

    def activation_out(self, x, random=False):
      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, int(self.out_size*self.out_sparsity)).indices] = 1
      return x_prime
 

    def pattern_complete(self, h_0=None):
      h = self.in_ if h_0 == None else h_0
      for iteration in range(self.pattern_complete_iterations):
        h = self.activation_in(F.linear(h, self.in_in))
      return h
    


    def hebbian_in_in(self):
      self.in_in_plastic += self.lmbda_in_in*torch.outer(self.in_, self.in_)

    def hebbian_out_in(self):
      self.out_in_plastic += self.lmbda_in_in*torch.outer(self.out, self.in_)


    def homeostasis_in_in(self):
      if self.homeostasis == 'none':
        pass
      elif self.homeostasis_in_in_type == 'bound':
        self.in_in_plastic = torch.clip(self.in_in_plastic, min=None, max=torch.min(self.max_post_in_in. self.max_pre_in_in))
        self.in_in = self.in_in_fixed + self.in_in_plastic
      elif self.homeostasis_in_in_type == 'renorm':
        # Calculate the total pre-connectivity for each neuron
        total_post_connectivity = torch.sum(self.in_in_plastic, dim=1)
        # Identify neurons that exceed the max pre-connectivity
        post_exceeding_mask = total_post_connectivity > self.max_post_in_in
        # Scale the connectivities of the exceeding neurons
        post_scaling_factors = torch.where(
            post_exceeding_mask,
            self.max_post_in_in / total_post_connectivity,
            torch.ones_like(total_post_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.in_in_plastic = self.in_in_plastic * post_scaling_factors.unsqueeze(1)


        # Calculate the total pre-connectivity for each neuron
        total_pre_connectivity = torch.sum(self.in_in_plastic, dim=0)
        # Identify neurons that exceed the max pre-connectivity
        pre_exceeding_mask = total_pre_connectivity > self.max_pre_in_in
        # Scale the connectivities of the exceeding neurons
        pre_scaling_factors = torch.where(
            pre_exceeding_mask,
            self.max_pre_in_in / total_pre_connectivity,
            torch.ones_like(total_pre_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.in_in_plastic = self.in_in_plastic * pre_scaling_factors


        self.in_in = self.in_in_fixed + self.in_in_plastic

      else:
        print("This type of homeostatic plasticity is not implemented")



    def homeostasis_out_in(self):
      if self.homeostasis_out_in_type == 'none':
        pass
      elif self.homeostasis_out_in_type == 'bound':
        self.out_in_plastic = torch.clip(self.out_in_plastic, min=None, max=torch.min(self.max_post_out_in. self.max_pre_out_in))
        self.out_in = self.out_in_fixed + self.out_in_plastic
      elif self.homeostasis_out_in_type == 'renorm':
        # Calculate the total pre-connectivity for each neuron
        total_post_connectivity = torch.sum(self.out_in_plastic, dim=1)
        # Identify neurons that exceed the max pre-connectivity
        post_exceeding_mask = total_post_connectivity > self.max_post_out_in
        # Scale the connectivities of the exceeding neurons
        post_scaling_factors = torch.where(
            post_exceeding_mask,
            self.max_post_out_in / total_post_connectivity,
            torch.ones_like(total_post_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.out_in_plastic = self.out_in_plastic * post_scaling_factors.unsqueeze(1)


        # Calculate the total pre-connectivity for each neuron
        total_pre_connectivity = torch.sum(self.out_in_plastic, dim=0)
        # Identify neurons that exceed the max pre-connectivity
        pre_exceeding_mask = total_pre_connectivity > self.max_pre_out_in
        # Scale the connectivities of the exceeding neurons
        pre_scaling_factors = torch.where(
            pre_exceeding_mask,
            self.max_pre_out_in / total_pre_connectivity,
            torch.ones_like(total_pre_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.out_in_plastic = self.out_in_plastic * pre_scaling_factors


        self.out_in = self.out_in_fixed + self.out_in_plastic

      else:
        print("This type of homeostatic plasticity is not implemented")



    def daily_reset(self):
      pass

    def init_network(self, net_params):

      #initialize network parameters
      for key, value in net_params.items():
        setattr(self, key, value)

      #define subnetworks
      self.in_ = torch.zeros((self.in_size))
      self.in_hat = torch.zeros((self.in_size))
      self.out = torch.zeros((self.out_size))

      #define connectivity
      self.in_in_sparsity_mask = torch.rand((self.in_size, self.in_size)) < self.in_in_sparsity
      self.in_in_fixed = nn.Linear(self.in_size, self.in_size, bias=False).weight.clone().detach()*self.in_in_g*self.in_in_sparsity_mask
      self.in_in_plastic = torch.zeros((self.in_size, self.in_size))
      self.in_in = self.in_in_fixed + self.in_in_plastic

      self.out_in_sparsity_mask = torch.rand((self.in_size, self.in_size)) < self.out_in_sparsity
      self.out_in_fixed = nn.Linear(self.out_size, self.in_size, bias=False).weight.clone().detach()*self.out_in_g*self.out_in_sparsity_mask
      self.out_in_plastic = torch.zeros((self.out_size, self.in_size))
      self.out_in = self.out_in_fixed + self.out_in_plastic     

      #initialize temporal variables
      self.time_index = 0

    def init_recordings(self, rec_params):
      self.activity_recordings = {}
      for region in rec_params["regions"]:
        self.activity_recordings[region] = np.array([getattr(self, region)])
      self.activity_recordings_rate = rec_params["rate_activity"]
      self.activity_recordings_time = np.array([])
      self.connectivity_recordings = {}
      for connection in rec_params["connections"]:
        self.connectivity_recordings[connection] = np.array([getattr(self, connection)])
      self.connectivity_recordings_time = np.array([])
      self.connectivity_recordings_rate = rec_params["rate_connectivity"]

    def record(self):
      if self.time_index%self.activity_recordings_rate == 0:
        for region in self.activity_recordings:
          layer_activity = getattr(self, region)
          self.activity_recordings[region] = np.append(self.activity_recordings[region], [deepcopy(layer_activity.detach().numpy())], axis=0)
          self.activity_recordings_time = np.append(self.activity_recordings_time, self.time_index)
      if self.time_index%self.connectivity_recordings_rate == 0:
        for connection in self.connectivity_recordings:
          connection_state = getattr(self, connection)
          self.connectivity_recordings[connection] = np.append(self.connectivity_recordings[connection], [deepcopy(connection_state.detach().numpy())], axis=0)
          self.connectivity_recordings_time = np.append(self.connectivity_recordings_time, self.time_index)