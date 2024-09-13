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

    def forward(self, input, test=False):
        output = []
      
        for timestep in range(input.shape[0]):

            self.in_hat = input[timestep]
            self.in_ =  self.pattern_complete(self.in_hat) if self.do_pattern_complete else self.activation_in(self.in_hat)
            
            self.out_hat =  F.linear(self.in_, self.out_in)
            self.out = self.activation_out(self.out_hat, random=(not self.forward_input))

            if self.learn_rec:
              self.hebbian_in_in()
              self.homeostasis_in_in()
            if self.learn_ff:
              self.hebbian_out_in()
              self.homeostasis_out_in()

            self.record()
            self.time_index += 1

            if test:
              output.append(self.out)
        
        if test:
          return torch.stack(output)


    def sleep(self, num_timesteps):
      for time in range(num_timesteps):
        x = torch.randn(self.in_.shape)
        self.in_ = self.pattern_complete(x, num_iterations=5)
        self.out_hat =  F.linear(self.in_, self.out_in)
        self.out = self.activation_out(self.out_hat, random=(not self.forward_input))
        self.hebbian_out_in()
        self.homeostasis_out_in()
        self.record()




    def activation_in(self, x, random=False):
      x = torch.randn(x.shape) if random else x + (1e-10 + torch.max(x) - torch.min(x))/100*torch.randn(x.shape)
      final_x = torch.zeros(x.shape)
      for region_index, region in enumerate(self.in_regions):
        x_prime = torch.zeros(len(region))
        top_indices = torch.topk(x[region], int(len(region)*self.in_sparsity[region_index])).indices
        x_prime[top_indices] = 1
        final_x[region]  = x_prime
      return final_x

    def activation_out(self, x, random=False):
      x = torch.randn(x.shape) if random else x + torch.abs(torch.min(x)/10)*torch.randn(x.shape)
      x_prime = torch.zeros(x.shape)
      x_prime[torch.topk(x, int(self.out_size*self.out_sparsity)).indices] = 1
      return x_prime

    def pattern_complete(self, h_0=None, num_iterations=None, depress_synapses=False):

      input_mask = torch.outer(h_0, h_0)
      depression_mask = torch.ones_like(self.in_in)
      aux_synapses = self.in_in.clone()
      h = self.in_ if h_0 == None else h_0
      h = self.activation_in(h)
      num_iterations = self.pattern_complete_iterations if num_iterations == None else num_iterations
      for iteration in range(num_iterations):
        h = self.activation_in(F.linear(h, depression_mask*aux_synapses))
        #h = self.activation_in(F.linear(h, self.in_in))
        if depress_synapses:
          delta_depression = self.depression_amplitude*torch.outer(h, h)
          depression_mask[~input_mask.bool()] = ((1 - self.depression_beta)*depression_mask - delta_depression)[~input_mask.bool()]
      return h
    
    def hebbian_in_in(self):
      self.in_in_plastic += self.lmbda_in_in*torch.outer(self.in_, self.in_)*self.hebb_dist_filter
      #self.in_in_plastic += self.lmbda_in_in*torch.outer(self.in_, self.in_)

    def hebbian_out_in(self):
      self.out_in_plastic += self.lmbda_out_in*torch.outer(self.out, self.in_)


    def homeostasis_in_in(self):
      def homeostasis_in_in_pre():
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
      
      def homeostasis_in_in_post():
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
        #self.in_in_plastic = self.in_in_plastic * post_scaling_factors.unsqueeze(1)
        self.in_in_plastic = self.in_in_plastic * post_scaling_factors
        self.in_in = self.in_in_fixed + self.in_in_plastic


      def homeostasis_in_in_mixed():
        # Calculate the total pre-connectivity for each neuron
        total_post_connectivity = torch.sum(self.in_in_plastic, dim=1)
        total_pre_connectivity = torch.sum(self.in_in_plastic, dim=0)
        total_av_connectivity = (total_pre_connectivity + total_post_connectivity.unsqueeze(1))/2
        max_mixed = (self.max_pre_in_in + self.max_post_in_in)/2


        pre_exceeding_mask = total_pre_connectivity > self.max_pre_in_in
        post_exceeding_mask = total_post_connectivity > self.max_post_in_in
        pre_post_exceeding_mask = pre_exceeding_mask.unsqueeze(0) & post_exceeding_mask.unsqueeze(1)



        pre_scaling_factors = torch.where(
            pre_exceeding_mask & ~pre_post_exceeding_mask,
            self.max_pre_in_in / total_pre_connectivity,
            torch.ones_like(self.in_in_plastic)
        )

        post_scaling_factors = torch.where(
            post_exceeding_mask.unsqueeze(1) & ~pre_post_exceeding_mask,
            self.max_post_in_in / total_post_connectivity,
            torch.ones_like(self.in_in_plastic)
        )

        pre_post_scaling_factors = torch.where(
            pre_post_exceeding_mask,
            max_mixed / total_av_connectivity,
            torch.ones_like(self.in_in_plastic)
        )

        self.in_in_plastic = self.in_in_plastic*pre_scaling_factors*post_scaling_factors*pre_post_scaling_factors
        self.in_in = self.in_in_fixed + self.in_in_plastic


      if self.homeostasis_in_in_type == 'none':
        pass
      elif self.homeostasis_in_in_type == 'bound':
        self.in_in_plastic = torch.clip(self.in_in_plastic, min=None, max=torch.min(self.max_post_in_in. self.max_pre_in_in))
        self.in_in = self.in_in_fixed + self.in_in_plastic
      elif self.homeostasis_in_in_type == 'renorm':
        homeostasis_in_in_mixed()
        '''
        if self.time_index%2 == 0:
          homeostasis_in_in_pre()
          homeostasis_in_in_post()
        else:
          homeostasis_in_in_post()
          homeostasis_in_in_pre()
        '''

      else:
        print("This type of homeostatic plasticity is not implemented")



    def homeostasis_out_in(self):
      def homeostasis_out_in_pre():
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
      
      def homeostasis_out_in_post():
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
        self.out_in = self.out_in_fixed + self.out_in_plastic


      if self.homeostasis_out_in_type == 'none':
        pass
      elif self.homeostasis_out_in_type == 'bound':
        self.out_in_plastic = torch.clip(self.out_in_plastic, min=None, max=torch.min(self.max_post_out_in. self.max_pre_out_in))
        self.out_in = self.out_in_fixed + self.out_in_plastic
        if self.time_index%2 == 0:
          homeostasis_out_in_pre()
          homeostasis_out_in_post()
        else:
          homeostasis_out_in_post()
          homeostasis_out_in_pre()




      else:
        print("This type of homeostatic plasticity is not implemented")



    def daily_reset(self):
      pass

    def init_network(self, net_params):

      #initialize network parameters
      for key, value in net_params.items():
        setattr(self, key, value)

      #define subnetworks
      self.in_hat = torch.zeros((self.in_size))
      self.in_ = torch.zeros((self.in_size))
      self.out = torch.zeros((self.out_size))
      self.out = torch.zeros((self.out_size))

      #define connectivity
      self.in_in_sparsity_mask = torch.rand((self.in_size, self.in_size)) < self.in_in_sparsity
      self.in_in_fixed = nn.Linear(self.in_size, self.in_size, bias=False).weight.clone().detach()*self.in_in_g*self.in_in_sparsity_mask
      self.in_in_plastic = torch.zeros((self.in_size, self.in_size))
      self.in_in = self.in_in_fixed + self.in_in_plastic

      self.out_in_sparsity_mask = torch.rand((self.out_size, self.in_size)) < self.out_in_sparsity
      self.out_in_fixed = nn.Linear(self.in_size, self.out_size, bias=False).weight.clone().detach()*self.out_in_g*self.out_in_sparsity_mask
      self.out_in_plastic = torch.zeros((self.out_size, self.in_size))
      self.out_in = self.out_in_fixed + self.out_in_plastic   

      self.hebb_dist_filter = torch.ones(self.in_in.shape) if self.hebb_distance_filter == None else self.init_distance_filter(self.hebb_distance_filter)

      #initialize temporal variables
      self.time_index = 0


    '''
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


    def init_distance_filter(self, radius):
      # Image size
      width, height = 28, 28

      # Define the pixel grid
      x, y = np.meshgrid(np.arange(width), np.arange(height))

      # Flatten the grid to create (784, 2) coordinates
      coords = np.stack([x.ravel(), y.ravel()], axis=1)

      # Compute the pairwise Euclidean distances
      distances = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))

      # Compute alpha using the equation alpha = -ln(0.01) / r
      alpha = -np.log(0.01) / radius

      # Apply the exponential decay function
      distance_tensor = np.exp(-alpha * distances)

      # The distance_tensor now has a shape of (784, 784)

      return distance_tensor