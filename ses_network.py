import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from collections import OrderedDict

class SESNetwork(nn.Module):
    def __init__(self, net_params, rec_params):

      super(SESNetwork, self).__init__()
      self.init_network(net_params)
      self.init_recordings(rec_params)

    def forward(self, input, test=False):

      self.daily_reset()
      self.process_input(input, test)


      if not test:
        self.sleep()

      self.baby_days += 1

    def process_input(self, input, test=False):
      for timestep in range(input.shape[0]):
        #forward sensory to hpc
        self.lec_hat = self.gamma_lec_sen*F.linear(input[timestep], self.lec_sen)
        self.lec = self.activation_lec(self.lec_hat)
        self.hpc[:self.lec_size] = self.lec


        #forward sensory and hpc to pfc
        self.pfc_hat = self.gamma_pfc_sen*input[timestep] + self.gamma_pfc_lec*F.linear(self.lec, self.pfc_lec) + self.gamma_pfc_mec*F.linear(self.mec, self.pfc_mec)

        pfc_hat_noise_std = self.pfc_hat[torch.abs(self.pfc_hat) != 0].min()/10
        self.pfc_hat = self.pfc_hat + torch.randn_like(self.pfc_hat) * pfc_hat_noise_std
        self.pfc = self.activation_pfc(self.pfc_hat)

        if not test:
          #store hpc pattern
          self.hebbian_hpc_hpc()
          #potentiate hpc to pfc
          self.hebbian_pfc_lec()

          if self.baby_days < self.total_baby_days:
            self.mec_hat = torch.zeros((self.mec_size))
            self.mec = torch.zeros((self.mec_size))
          else:
            self.pfc = self.pattern_complete_pfc()
            self.pfc = self.activation_pfc(self.pfc)
            self.mec_hat = self.gamma_mec_pfc*F.linear(self.pfc, self.mec_pfc)
            self.mec = self.activation_mec(self.mec_hat)
        else:
            self.pfc = self.pattern_complete_pfc()
            self.pfc = self.activation_pfc(self.pfc)

        #forward mec activity to hpc
        self.hpc[self.lec_size:] = self.mec
        self.time_index += 1

        self.record()


    def sleep(self):
      for timestep in range(self.sleep_time):
        h_0 = torch.bernoulli(self.random_hpc_sparsity*torch.ones(self.hpc_size))
        self.hpc = self.pattern_complete_hpc(h_0)
        self.hpc = self.activation_hpc(self.hpc)
        self.pfc_hat = self.gamma_pfc_lec*F.linear(self.hpc[:self.lec_size], self.pfc_lec) + self.gamma_pfc_mec*F.linear(self.hpc[self.lec_size:], self.pfc_mec)
        self.pfc = self.activation_pfc(self.pfc_hat)

        self.hebbian_pfc_pfc()
        self.homeostasis_pfc_pfc()

        #update mec neurons (if not baby anymore)
        if self.baby_days < self.total_baby_days:
          self.mec_hat = torch.zeros((self.mec_size))
          self.mec = torch.zeros((self.mec_size))
        else:
          h_0 = torch.bernoulli(self.random_pfc_sparsity*torch.ones(self.pfc_size))
          self.pfc = self.pattern_complete_pfc(h_0, top_k=50)
          self.pfc = self.activation_pfc(self.pfc, top_k=50)
          #self.mec_hat = self.gamma_mec_pfc*torch.matmul(self.mec_pfc, self.pfc)
          self.mec_hat = self.gamma_mec_pfc*F.linear(self.pfc, self.mec_pfc)
          self.mec = self.activation_mec(self.mec_hat, top_k=5)
          self.hebbian_mec_pfc()
          self.homeostasis_mec_pfc()
        self.time_index += 1
        self.record()

    def activation_lec(self, x):
      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, int(self.lec_size*self.lec_sparsity)).indices] = 1
      return x_prime

    def activation_mec(self, x, top_k=None):
      top_k = int(self.mec_size*self.mec_sparsity) if top_k==None else top_k
      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, top_k).indices] = 1
      return x_prime

    def activation_hpc(self, x):
      x_prime = torch.zeros(x.shape)
      if self.baby_days < self.total_baby_days:
        x_prime[torch.topk(x, int(self.lec_size*self.lec_sparsity)).indices] = 1
      else:
        x_prime[torch.topk(x, int(self.lec_size*self.lec_sparsity + self.mec_size*self.mec_sparsity)).indices] = 1
      return x_prime

    def activation_pfc(self, x, top_k=None):
      top_k = int(self.pfc_size*self.pfc_sparsity) if top_k==None else top_k

      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, top_k).indices] = 1

      return x_prime

    def pattern_complete_hpc(self, h_0=None):
      h = self.pfc_hpc(self.pfc) if h_0 == None else h_0
      for iteration in range(self.pattern_complete_iterations):
        h = self.beta_hpc*h + self.activation_hpc(self.gamma_hpc_hpc*F.linear(h, self.hpc_hpc))
      return h

    def pattern_complete_pfc(self, h_0=None, top_k=None, num_iterations=None):
      h = self.pfc if h_0 == None else h_0
      num_iterations = self.pattern_complete_iterations if num_iterations == None else num_iterations
      for iteration in range(num_iterations):
        h = self.beta_pfc*h + (1 - self.beta_pfc)*self.activation_pfc(self.gamma_pfc_pfc*F.linear(h, self.pfc_pfc), top_k)
      return h

    def hebbian_hpc_hpc(self):
      self.hpc_hpc += self.lmbda_hpc_hpc*torch.outer(self.hpc, self.hpc)
      self.hpc_hpc.fill_diagonal_(0.)

    def hebbian_pfc_pfc(self):
      self.pfc_pfc += self.lmbda_pfc_pfc*torch.outer(self.pfc, self.pfc)
      self.pfc_pfc.fill_diagonal_(0.)

    def hebbian_pfc_lec(self):
      self.pfc_lec += self.lmbda_pfc_lec*torch.outer(self.pfc, self.lec)

    def hebbian_mec_pfc(self):
      self.mec_pfc += self.lmbda_mec_pfc*torch.outer(self.mec, self.pfc)

    def hebbian_pfc_mec(self):
      self.pfc_mec += self.lmbda_pfc_mec*torch.outer(self.mec, self.pfc)

    def homeostasis_pfc_pfc(self):
      if self.homeostasis == 'none':
        pass

      elif self.homeostasis == 'bound':
        self.pfc_pfc = torch.clip(self.pfc_pfc, min=None, max=self.max_connectivity/100)

      elif self.homeostasis == 'renorm':


        # Calculate the total pre-connectivity for each neuron
        total_post_connectivity = torch.sum(self.pfc_pfc, dim=1)
        # Identify neurons that exceed the max pre-connectivity
        post_exceeding_mask = total_post_connectivity > self.max_post_pfc_pfc_connectivity
        # Scale the connectivities of the exceeding neurons
        post_scaling_factors = torch.where(
            post_exceeding_mask,
            self.max_post_pfc_pfc_connectivity / total_post_connectivity,
            torch.ones_like(total_post_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.pfc_pfc = self.pfc_pfc * post_scaling_factors.unsqueeze(1)


        # Calculate the total pre-connectivity for each neuron
        total_pre_connectivity = torch.sum(self.pfc_pfc, dim=0)
        # Identify neurons that exceed the max pre-connectivity
        pre_exceeding_mask = total_pre_connectivity > self.max_pre_pfc_pfc_connectivity
        # Scale the connectivities of the exceeding neurons
        pre_scaling_factors = torch.where(
            pre_exceeding_mask,
            self.max_pre_pfc_pfc_connectivity / total_pre_connectivity,
            torch.ones_like(total_pre_connectivity)
        )
        # Apply the scaling factors to the connectivity matrix
        self.pfc_pfc = self.pfc_pfc * pre_scaling_factors

      else:
        print("This type of homeostatic plasticity is not implemented")

    def homeostasis_mec_pfc(self):
      # Calculate the total pre-connectivity for each neuron
      total_post_connectivity = torch.sum(self.mec_pfc, dim=1)
      # Identify neurons that exceed the max pre-connectivity
      post_exceeding_mask = total_post_connectivity > self.max_post_mec_pfc_connectivity
      # Scale the connectivities of the exceeding neurons
      post_scaling_factors = torch.where(
          post_exceeding_mask,
          self.max_post_mec_pfc_connectivity / total_post_connectivity,
          torch.ones_like(total_post_connectivity)
      )
      # Apply the scaling factors to the connectivity matrix
      self.mec_pfc = self.mec_pfc * post_scaling_factors.unsqueeze(1)


      # Calculate the total pre-connectivity for each neuron
      total_pre_connectivity = torch.sum(self.mec_pfc, dim=0)
      # Identify neurons that exceed the max pre-connectivity
      pre_exceeding_mask = total_pre_connectivity > self.max_pre_mec_pfc_connectivity
      # Scale the connectivities of the exceeding neurons
      pre_scaling_factors = torch.where(
          pre_exceeding_mask,
          self.max_pre_mec_pfc_connectivity / total_pre_connectivity,
          torch.ones_like(total_pre_connectivity)
      )
      # Apply the scaling factors to the connectivity matrix
      self.mec_pfc = self.mec_pfc * pre_scaling_factors


    def daily_reset(self):
      self.hpc_hpc = torch.zeros((self.hpc_size, self.hpc_size))
      self.hpc = torch.zeros((self.hpc_size))
      self.pfc_lec = torch.zeros((self.pfc_size, self.lec_size))

    def init_network(self, net_params):

      #initialize network parameters
      for key, value in net_params.items():
        setattr(self, key, value)

      self.beta_hpc = np.exp(-1/self.tau_hpc)
      self.beta_pfc = np.exp(-1/self.tau_pfc)
      self.hpc_pattern_width = int(np.sqrt(self.hpc_size))
      self.pfc_pattern_width = int(np.sqrt(self.pfc_size))


      #define subnetworks
      self.lec = torch.zeros((self.lec_size))
      self.mec_hat = torch.zeros((self.mec_size))
      self.mec = torch.zeros((self.mec_size))
      self.hpc = torch.zeros((self.hpc_size))
      self.pfc = torch.zeros((self.pfc_size))
      self.pfc_hat = torch.zeros((self.pfc_size))

      #define connectivity
      self.hpc_hpc_sparsity_mask = torch.rand((self.hpc_size, self.hpc_size)) < self.hpc_hpc_sparsity
      self.hpc_hpc = nn.Linear(self.hpc_size, self.hpc_size, bias=False).weight.clone().detach()*self.hpc_hpc_g*self.hpc_hpc_sparsity_mask
      self.pfc_pfc_sparsity_mask = torch.rand((self.pfc_size, self.pfc_size)) < self.pfc_pfc_sparsity
      self.pfc_pfc = nn.Linear(self.pfc_size, self.pfc_size, bias=False).weight.clone().detach()*self.pfc_pfc_g*self.pfc_pfc_sparsity_mask
      self.pfc_lec = torch.zeros((self.pfc_size, self.lec_size))
      self.lec_sen = nn.Linear(self.sen_size, self.lec_size, bias=False).weight.clone().detach()
      self.pfc_sen = torch.eye(self.pfc_size)
      self.pfc_mec = torch.zeros((self.pfc_size, self.mec_size))
      self.mec_pfc = nn.Linear(self.pfc_size, self.mec_size, bias=False).weight.clone().detach()


      #define homeostatic params
      self.max_pre_pfc_pfc_connectivity = (1/self.eta_pre_pfc_pfc)*self.max_connectivity if self.eta_pre_pfc_pfc != 0 else np.inf
      self.max_post_pfc_pfc_connectivity = (1/(1 - self.eta_pre_pfc_pfc))*self.max_connectivity if self.eta_pre_pfc_pfc != 1 else np.inf
      self.max_pre_mec_pfc_connectivity = (1/self.eta_pre_mec_pfc)*self.max_connectivity if self.eta_pre_mec_pfc != 0 else np.inf
      self.max_post_mec_pfc_connectivity = (1/(1 - self.eta_pre_mec_pfc))*self.max_connectivity if self.eta_pre_mec_pfc != 1 else np.inf

      #initialize temporal variables
      self.time_index = 0
      self.baby_days = 0

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
      if self.time_index%self.activity_recordings_rate == 1:
        for region in self.activity_recordings:
          layer_activity = getattr(self, region)
          self.activity_recordings[region] = np.append(self.activity_recordings[region], [deepcopy(layer_activity.detach().numpy())], axis=0)
          self.activity_recordings_time = np.append(self.activity_recordings_time, self.time_index)
      if self.time_index%self.connectivity_recordings_rate == 0:
        for connection in self.connectivity_recordings:
          connection_state = getattr(self, connection)
          self.connectivity_recordings[connection] = np.append(self.connectivity_recordings[connection], [deepcopy(connection_state.detach().numpy())], axis=0)
          self.connectivity_recordings_time = np.append(self.connectivity_recordings_time, self.time_index)
    '''
    def init_recordings(self, rec_params):
      self.activity_recordings = {}
      for region in rec_params["regions"]:
        self.activity_recordings[region] = [getattr(self, region)]
      self.activity_recordings_rate = rec_params["rate_activity"]
      self.activity_recordings_time = []
      self.connectivity_recordings = {}
      for connection in rec_params["connections"]:
        self.connectivity_recordings[connection] = [getattr(self, connection)]
      self.connectivity_recordings_time = []
      self.connectivity_recordings_rate = rec_params["rate_connectivity"]

    def record(self):
      if self.time_index%self.activity_recordings_rate == 0:
        for region in self.activity_recordings:
          layer_activity = getattr(self, region)
          self.activity_recordings[region].append(deepcopy(layer_activity.detach().numpy()))
          #self.activity_recordings[region] = np.append(self.activity_recordings[region], [deepcopy(layer_activity.detach().numpy())], axis=0)
          self.activity_recordings_time.append(self.time_index)
      if self.time_index%self.connectivity_recordings_rate == 0:
        for connection in self.connectivity_recordings:
          connection_state = getattr(self, connection)
          self.connectivity_recordings[connection].append(deepcopy(connection_state.detach().numpy()))
          self.connectivity_recordings_time.append(self.time_index)

    def recordings_to_np(self):
      for region in self.activity_recordings:
        self.activity_recordings[region] = np.array(self.activity_recordings[region])
      for connection in self.connectivity_recordings:
        self.connectivity_recordings[connection] = np.array(self.connectivity_recordings[connection])


    def reset_recordings(self):
      for region in self.activity_recordings:
        self.activity_recordings[region] = np.array(self.activity_recordings[region])
      for connection in self.connectivity_recordings:
        self.connectivity_recordings[connection] = np.array(self.connectivity_recordings[connection])

    '''