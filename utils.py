import numpy as np
import cv2
import torch

import itertools
import random

from collections import OrderedDict




def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def get_cond_matrix(latent_space, network):
  num_subs = len(latent_space.sub_index_to_neuron_index)
  sim_cond_matrix = np.zeros((num_subs, num_subs))
  th_cond_matrix = np.zeros((num_subs, num_subs))
  for conditioned_sub_index, ((conditioned_latent, conditioned_sub), conditioned_neuron_index) in enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
    for condition_sub_index, ((condition_latent, condition_sub), condition_neuron_index) in  enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
      sim_cond_matrix[conditioned_sub_index][condition_sub_index] = torch.mean(network.pfc_pfc[conditioned_neuron_index][:, condition_neuron_index])
      if conditioned_latent != condition_latent:
        label = [0, 0]
        label[conditioned_latent] = conditioned_sub
        label[condition_latent] = condition_sub
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = latent_space.label_to_probs[tuple(label)]/latent_space.sub_index_to_marginal[condition_sub_index]

      elif conditioned_sub == condition_sub:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 1

      else:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0
  return sim_cond_matrix, th_cond_matrix

def get_sample_from_num_swaps(x_0, num_swaps):
  x = x_0.clone().detach()
  #get on and off index
  on_index = x_0.nonzero().squeeze(1)
  off_index = (x_0 ==0).nonzero().squeeze(1)
  #choose at random num_flips indices
  flip_off = on_index[torch.randperm(len(on_index))[:num_swaps]]
  flip_on = off_index[torch.randperm(len(off_index))[:num_swaps]]
  #flip on to off and off to on
  x[flip_off] = 0
  x[flip_on] = 1
  return x




class LatentSpace():
    def __init__(self, num, total_sizes, act_sizes, dims, prob_list, random_neurons=False):
      self.num_latents = num
      self.dims = dims
      self.total_sizes = total_sizes
      self.total_size = sum(total_sizes)
      self.act_sizes = act_sizes
      self.random_neurons = random_neurons
      self.latent_patterns = [[self.get_sub_latent(latent, sub_dim) for sub_dim in range(self.dims[latent])] for latent in range(self.num_latents)]
      self.sub_index_to_neuron_index = [self.latent_patterns[latent][sub_dim].nonzero().squeeze(1).detach().numpy() + sum(self.total_sizes[:latent]) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.sub_index_to_latent_sub = [(latent, sub_dim) for latent in range(self.num_latents) for sub_dim in range(self.dims[latent])]
      self.index_to_label = list(itertools.product(*[[i for i in range(dim)] for dim in self.dims]))
      self.label_to_index = {label: index for index, label in enumerate(self.index_to_label)}
      self.label_to_neurons = OrderedDict({label: self.get_neurons_from_label(label) for label in self.index_to_label})
      print(prob_list)
      self.label_to_probs = OrderedDict({label: prob for label, prob in zip(self.index_to_label, prob_list)})
      self.sub_index_to_marginal = [self.get_marginal(latent, sub) for (latent, sub) in self.sub_index_to_latent_sub]

    def get_sub_latent(self, latent, sub_dim):
      if self.random_neurons:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[torch.randperm(self.total_sizes[latent])[:self.act_sizes[latent]]] = 1
      else:
        sub_latent = torch.zeros((self.total_sizes[latent]))
        sub_latent[sub_dim*self.act_sizes[latent]:(sub_dim+1)*self.act_sizes[latent]] = 1
      return sub_latent

    def get_neurons_from_label(self, label):
        return torch.cat(tuple(self.latent_patterns[latent][index] for latent, index in enumerate(label)))

    def get_marginal(self, latent, sub):
      marginal = 0
      for label, prob in self.label_to_probs.items():
        if label[latent] == sub:
          marginal += prob
      return marginal

    def sample(self):
      return random.choices(list(self.label_to_neurons.items()), weights=list(self.label_to_probs.values()))[0]
    

class SatelliteSpace():
  def __init__(self, neoctx_size):
    self.neoctx_size = neoctx_size
    self.total_size = self.neoctx_size
    self.names = ["gavan", "volar", "motar", "nivex", "sorex", "denor", "sopra", "funda", "bacta", "gondo", "malar", "benin", "colar", "nodon", "praxa"]
    self.alpha_names = ["gavan", "volar", "motar", "nivex", "sorex"]
    self.beta_names = ["denor", "sopra", "funda", "bacta", "gondo"]
    self.gamma_names = ["malar", "benin", "colar", "nodon", "praxa"]
    self.classes = ["alpha", "beta", "gamma"]
    self.num_classes = len(self.classes)
    self.num_names = len(self.names)
    self.alpha_prototype = []
    self.name_to_class = {
        name: class_
        for class_names, class_ in zip([self.alpha_names, self.beta_names, self.gamma_names], self.classes)
        for name in class_names
    }
    self.num_attributes = 5
    self.name_to_attributes = {
        name: self.get_attributes_from_name(name)
        for name in self.names
    }
    self.name_to_neural = {
        name: self.get_neural_from_name(name)
        for name in self.names
    }
    self.element_to_neural = self.get_elements_to_neural()

  def get_elements_to_neural(self):
      elements_to_neural = {}
      attributes = ["name", "class", "attribute1", "attribute2", "attribute3", "attribute4", "attribute5"]
      elements_per_attribute = [15, 3, 3, 6, 6, 6, 6]
      n_act = 25
      starting = 0
      for attribute_index, attribute in enumerate(attributes):
        neurals = []
        for element in range(elements_per_attribute[attribute_index]):
          neural = torch.zeros(self.neoctx_size)
          neural[starting:starting+n_act] = 1
          starting += n_act
          neurals.append(neural)
        elements_to_neural[attribute] = neurals
      return elements_to_neural
  def get_attributes_from_name(self, name):
    #make a tensor num of attributes (5) times num_classes*2 (6)
    attributes_0 = torch.zeros((1, self.num_classes))
    class_index = self.classes.index(self.name_to_class[name])
    attributes_0[0, class_index] = 1
    attributes_1 = torch.zeros((self.num_attributes - 1, self.num_classes*2))
    #get name index (0 to 14)
    name_index = self.names.index(name)
    #get name class index --> this is within class index, tells you which of the elements
    name_class_index = name_index%(self.num_attributes)
    for att_num in range(self.num_attributes - 1):
      if name_class_index - 1 == att_num:
        attributes_1[att_num, class_index+self.num_classes] = 1
      else:
        attributes_1[att_num, class_index] = 1


    #attributes_0 = torch.zeros_like(attributes_0)
    attributes = torch.cat([attributes_0.flatten(), attributes_1.flatten()])

    return attributes.flatten()

  def get_neural_from_name(self, name):
    neural = torch.zeros((self.neoctx_size))
    neural_name = torch.nn.functional.one_hot(torch.tensor(self.names.index(name)), num_classes=self.num_names)
    neural_class = torch.nn.functional.one_hot(torch.tensor(self.classes.index(self.name_to_class[name])), num_classes=self.num_classes)
    neural_attributes = self.name_to_attributes[name]
    min_neural = torch.cat((neural_name, neural_class, neural_attributes))
    min_neural_size = len(min_neural)
    min_neural_repeats = self.neoctx_size//min_neural_size
    if min_neural_repeats == 0:
      print("neoctx is not big enough for minimal neural representation, should be at least {}".format(min_neural_size))
    neural[:min_neural_size*min_neural_repeats] = min_neural.repeat_interleave(min_neural_repeats)
    return neural

  def sample(self):
    name = random.choice(self.names)
    label = self.names.index(name)
    return label, self.name_to_neural[name]
    
def get_cos_sim_torch(x1, x2):
  return torch.dot(x1, x2)/(torch.norm(x1)*torch.norm(x2))
def get_cos_sim_np(x1, x2):
  return np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2))


def get_cond_matrix(latent_space, network):
  num_subs = len(latent_space.sub_index_to_neuron_index)
  sim_cond_matrix = np.zeros((num_subs, num_subs))
  th_cond_matrix = np.zeros((num_subs, num_subs))
  for conditioned_sub_index, ((conditioned_latent, conditioned_sub), conditioned_neuron_index) in enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
    for condition_sub_index, ((condition_latent, condition_sub), condition_neuron_index) in  enumerate(zip(latent_space.sub_index_to_latent_sub, latent_space.sub_index_to_neuron_index)):
      sim_cond_matrix[conditioned_sub_index][condition_sub_index] = torch.mean(network.pfc_pfc[conditioned_neuron_index][:, condition_neuron_index])
      if conditioned_latent != condition_latent:
        label = [0, 0]
        label[conditioned_latent] = conditioned_sub
        label[condition_latent] = condition_sub
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = latent_space.label_to_probs[tuple(label)]/latent_space.sub_index_to_marginal[condition_sub_index]

      elif conditioned_sub == condition_sub:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 1

      else:
        th_cond_matrix[conditioned_sub_index][condition_sub_index] = 0
  return sim_cond_matrix, th_cond_matrix

def get_sample_from_num_swaps(x_0, num_swaps):
  x = x_0.clone().detach()
  #get on and off index
  on_index = x_0.nonzero().squeeze(1)
  off_index = (x_0 ==0).nonzero().squeeze(1)
  #choose at random num_flips indices
  flip_off = on_index[torch.randperm(len(on_index))[:num_swaps]]
  flip_on = off_index[torch.randperm(len(off_index))[:num_swaps]]
  #flip on to off and off to on
  x[flip_off] = 0
  x[flip_on] = 1
  return x



def make_input(num_days, day_length, mean_duration, fixed_duration, num_swaps, latent_space, satellite=False):
  def get_partial_circle():
    delta_theta = 45
    image = np.zeros((40, 25), dtype=np.uint8)

    # Define the circle parameters
    x_center, y_center = 15, 20  # Center coordinates
    radius = 7  # Radius
    thickness = 2 # Thickness

    # Define the start and end angles for the partial circle
    theta = np.random.uniform(0, 360)
    start_angle = theta - delta_theta/2
    end_angle = theta + delta_theta/2


    # Draw the partial circle on the image
    color = 255  # White color for grayscale
    cv2.ellipse(image, (x_center, y_center), (radius, radius), 0, -start_angle, -end_angle, color, thickness)

    return int(theta), torch.tensor(image.flatten())

  def get_sample_from_err_rate(pattern):
    p_size = len(pattern)
    error_index = torch.randperm(p_size)[:int(p_size*error_rate)]
    pattern[error_index] = pattern[error_index]^ torch.ones_like(pattern[error_index])
    return pattern

  def get_sample_from_num_swaps(x_0, num_swaps):
    x = x_0.clone().detach()
    #get on and off index
    on_index = x_0.nonzero().squeeze(1)
    off_index = (x_0 ==0).nonzero().squeeze(1)
    #choose at random num_flips indices
    flip_off = on_index[torch.randperm(len(on_index))[:num_swaps]]
    flip_on = off_index[torch.randperm(len(off_index))[:num_swaps]]
    #flip on to off and off to on
    x[flip_off] = 0
    x[flip_on] = 1
    return x

  #initialize input tensor
  #input = torch.zeros((num_days, day_length, sum(latent_space.total_sizes)))
  input = torch.zeros((num_days, day_length, latent_space.total_size))
  input_latents = torch.zeros((num_days, day_length), dtype=torch.int32)

  #create input from noisy patterns
  for day in range(num_days):
    day_timestep = 0
    while day_timestep < day_length:
      #pattern_duration = pattern_duration if (day_timestep + pattern_duration <= day_length) else day_length - day_timestep
      pattern_duration = mean_duration if fixed_duration else int(torch.poisson(mean_duration*torch.ones(1))[0])
      if satellite:
        latent_index, pattern = latent_space.sample()
        input_latents[day, day_timestep:day_timestep+pattern_duration] = latent_index
      else:
        label, pattern = latent_space.sample()
        input_latents[day, day_timestep:day_timestep+pattern_duration] = latent_space.label_to_index[label]
      #label, pattern =  get_partial_circle()
      #input_latents[day, day_timestep:day_timestep+pattern_duration] = label
      input[day, day_timestep:(day_timestep+pattern_duration)] = get_sample_from_num_swaps(pattern, num_swaps)
      day_timestep += pattern_duration

  return input, input_latents
