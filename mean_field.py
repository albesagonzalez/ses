import numpy as np
import matplotlib.pyplot as plt



def get_attribute_from_neuron(neuron, latent):
  cum_count = 0
  for attribute, size in enumerate(latent.total_sizes):
    if neuron < cum_count + size:
      return attribute
    cum_count += size

def get_starting_neuron_from_attribute(attriute, latent):
  cum_count = 0
  for size in latent.total_sizes[:attriute]:
    cum_count += size
  return cum_count

def get_original_probs(element_i, element_j):
   if element_i == 0 and element_j == 0:
      p_11, p_10, p_01, p_00 = 0.5, 0, 0, 0.5

   if element_i == 0 and element_j == 1:
      p_11, p_10, p_01, p_00 = 0, 0.5, 0.5, 0

   if element_i == 0 and element_j == 2:
      p_11, p_10, p_01, p_00 = 0.5*0.8, 0.5*0.2, 0.5*0.8, 0.5*0.2

   if element_i == 0 and element_j == 3:
      p_11, p_10, p_01, p_00 = 0.5*0.2, 0.5*0.8, 0.5*0.2, 0.5*0.8

   if element_i == 1 and element_j == 0:
      p_11, p_10, p_01, p_00 = 0, 0.5, 0.5, 0

   if element_i == 1 and element_j == 1:
      p_11, p_10, p_01, p_00 = 0.5, 0, 0, 0.5

   if element_i == 1 and element_j == 2:
      p_11, p_10, p_01, p_00 = 0.5*0.8, 0.5*0.2, 0.5*0.8, 0.5*0.2

   if element_i == 1 and element_j == 3:
      p_11, p_10, p_01, p_00 = 0.5*0.2, 0.5*0.8, 0.5*0.2, 0.5*0.8

   if element_i == 2 and element_j == 0:
      p_11, p_10, p_01, p_00 = 0.5*0.8, 0.5*0.2, 0.5*0.8, 0.5*0.2

   if element_i == 2 and element_j == 1:
      p_11, p_10, p_01, p_00 = 0.5*0.8, 0.5*0.2, 0.5*0.8, 0.5*0.2

   if element_i == 2 and element_j == 2:
      p_11, p_10, p_01, p_00 = 0.8, 0, 0, 0.2

   if element_i == 2 and element_j == 3:
      p_11, p_10, p_01, p_00 = 0, 0.8, 0.2, 0

   if element_i == 3 and element_j == 0:
      p_11, p_10, p_01, p_00 = 0.5*0.2, 0.5*0.8, 0.5*0.2, 0.5*0.8

   if element_i == 3 and element_j == 1:
      p_11, p_10, p_01, p_00 = 0.5*0.2, 0.5*0.8, 0.5*0.2, 0.5*0.8

   if element_i == 3 and element_j == 2:
      p_11, p_10, p_01, p_00 = 0., 0.2, 0.8, 0.

   if element_i == 3 and element_j == 3:
      p_11, p_10, p_01, p_00 = 0.2, 0, 0, 0.8

   return p_11, p_10, p_01, p_00


def get_swap_probs(p0_xi_xj_1_1,  p0_xi_xj_1_0, p0_xi_xj_0_1, p0_xi_xj_0_0, K, N_swap, N):
    """
    Calculate the total probability of two neurons being active.

    Parameters:
    p0_xi_xj_1_1 (float): p0(x_i = 1, x_j = 1)
    p0_xi_xj_0_1 (float): p0(x_i = 0, x_j = 1)
    p0_xi_xj_0_0 (float): p0(x_i = 0, x_j = 0)
    K (int): The number of neurons
    N_swap (int): The number of swaps
    N (int): The total number of neurons

    Returns:
    float: The total probability p(x_i = 1, x_j = 1)
    """

    # First term
    term1 = ((K - N_swap / 2) * (K - N_swap / 2 - 1)) / (K * (K - 1))

    # Second term
    term2 = ((K - N_swap / 2) / K) * (N_swap / (N - K))


    # Third term
    term3 = (N_swap/2 / (N - K)) * ((N_swap/2 - 1) / (N - K - 1))

    # Total probability
    total_prob = term1*p0_xi_xj_1_1 + term2*(p0_xi_xj_1_0 + p0_xi_xj_0_1) + term3*p0_xi_xj_0_0

    return total_prob


def get_swap_marginal(K, N_swap, N, p_0_i):
    """
    Calculate the value of p(i) based on the given parameters.

    Parameters:
    K (int or float): Number of neurons in the subset
    N_swap (int or float): The number of swaps
    N (int or float): Total number of neurons
    p_0_i (float): The initial probability for neuron i

    Returns:
    float: The value of p(i)
    """
    term1 = ((K - N_swap / 2) / K) * ((K - N_swap / 2 - 1) / (K - 1)) * p_0_i
    term2 = (N_swap / (N - K)) * (1 - p_0_i)

    p_i = term1 + term2
    print(p_i)
    return p_i
'''
def get_probs(latent_space, input_size, network_sparsity):
    p_ij_latent = np.array(list(latent_space.label_to_probs.values())).reshape((latent_space.dims[0], latent_space.dims[1]))

    p = {}
    p["ij"] = np.zeros((input_size, input_size))
    for neuron_i in range(input_size):
        for neuron_j in range(input_size):
          attribute_i = get_attribute_from_neuron(neuron_i, latent_space)
          attribute_j = get_attribute_from_neuron(neuron_j, latent_space)
          element_i = (neuron_i - get_starting_neuron_from_attribute(attribute_i, latent_space))//latent_space.act_sizes[attribute_i]
          element_j = (neuron_j - get_starting_neuron_from_attribute(attribute_j, latent_space))//latent_space.act_sizes[attribute_j]
          if attribute_i ==  attribute_j:
            if element_i != element_j:
              p["ij"][neuron_i, neuron_j] = 0
            if element_i == element_j:
              if attribute_i == 0:
                p["ij"][neuron_i, neuron_j] = np.sum(p_ij_latent[element_i])
              else:
                p["ij"][neuron_i, neuron_j] = np.sum(p_ij_latent[:, element_i])
          else:
            (element_0, element_1) = (element_i, element_j) if attribute_i == 0 else (element_j, element_i)
            p["ij"][neuron_i, neuron_j] = p_ij_latent[element_0, element_1]
        
        p["i"] = network_sparsity*np.ones((input_size))
        p["j"] = network_sparsity*np.ones((input_size))
        p["i"][100:150] = 0.8
        p["i"][150:200] = 0.2
        p["j"][100:150] = 0.8
        p["j"][150:200] = 0.2

    return p
'''


def get_probs(latent_space, input_size, network_sparsity, K, N_swap, N):
    p_ij_latent = np.array(list(latent_space.label_to_probs.values())).reshape((latent_space.dims[0], latent_space.dims[1]))

    p = {}
    p["ij"] = np.zeros((input_size, input_size))
    for neuron_i in range(input_size):
        for neuron_j in range(input_size):
          attribute_i = get_attribute_from_neuron(neuron_i, latent_space)
          attribute_j = get_attribute_from_neuron(neuron_j, latent_space)
          element_i = (neuron_i - get_starting_neuron_from_attribute(attribute_i, latent_space))//latent_space.act_sizes[attribute_i]
          element_j = (neuron_j - get_starting_neuron_from_attribute(attribute_j, latent_space))//latent_space.act_sizes[attribute_j]
          element_i, element_j = element_i + 2*attribute_i, element_j + 2*attribute_j
          p11, p10, p01, p00 = get_original_probs(element_i, element_j)
          p["ij"][neuron_i][neuron_j] = get_swap_probs(p11, p10, p01, p00, K, N_swap, N)
        
    p["i"] = network_sparsity*np.ones((input_size))
    p["j"] = network_sparsity*np.ones((input_size))
    p["i"][:100] = get_swap_marginal(K, N_swap, N, 0.5)
    p["j"][:100] = get_swap_marginal(K, N_swap, N, 0.5)
    p["i"][100:150] = get_swap_marginal(K, N_swap, N, 0.8)
    p["i"][150:200] =get_swap_marginal(K, N_swap, N, 0.2)
    p["j"][100:150] = get_swap_marginal(K, N_swap, N, 0.8)
    p["j"][150:200] =  get_swap_marginal(K, N_swap, N, 0.2)

    return p

def get_mean_field_solution(t, post_i, pre_j, i, sp, p, only_vars=False):

    T_pre_free = sp["w_pre_max"]/(sp["K_post"]*sp["lmbda"]*p["j"][pre_j])
    T_post_free = sp["w_post_max"]/(sp["K_pre"]*sp["lmbda"]*p["i"][post_i])

    w_0 = i["w"]
    S_pre_0 = i["S_pre"]
    S_post_0 = i["S_post"]
    if T_pre_free < T_post_free:
      if t < T_pre_free:
        w = sp["lmbda"]*p["ij"][post_i, pre_j]*t
        tau_w = 0
        fp_w = None
        S_pre = S_pre_0 + sp["lmbda"]*sp["K_post"]*p["j"][pre_j]*t
        S_post = S_post_0 + sp["lmbda"]*sp["K_pre"]*p["i"][post_i]*t
      else:

        S_post_free_inf = S_post_0 + sp["lmbda"]*sp["K_pre"]*p["i"][post_i]*T_pre_free
        S_post_inf = sp["w_pre_max"]*sp["K_post"]*p["i"][post_i]/(sp["K_pre"]*p["j"][pre_j])
        tau_post = sp["w_pre_max"]/(sp["lmbda"]*sp["K_pre"]*p["j"][pre_j])
        #S_post_inf = sp["w_pre_max"]*sp["K_post"]*p["i"][post_i]/(sp["K_pre"]*0.3)
        #tau_post = sp["w_pre_max"]/(sp["lmbda"]*sp["K_pre"]*0.3)

        T_post_cond = -tau_post*np.log((S_post_inf - sp["w_pre_max"])/(S_post_inf - S_post_free_inf)) if (S_post_inf - sp["w_post_max"] > 0) and (S_post_free_inf != S_post_inf) else np.inf
        w_free_inf = sp["lmbda"]*p["ij"][post_i, pre_j]*T_pre_free
        tau_w = sp["w_pre_max"]/(sp["lmbda"]*sp["K_post"]*p["j"][pre_j])
        fp_w = sp["w_pre_max"]*p["ij"][post_i, pre_j]/(sp["K_post"]*p["j"][pre_j])
        if (t < T_post_cond):
            S_pre = sp["w_pre_max"]
            beta_post = 1 - np.exp(-(t - T_pre_free)/tau_post)
            S_post = (1 - beta_post)*S_post_free_inf + beta_post*S_post_inf
            beta_w = 1 - np.exp(-(t - T_pre_free)/tau_w)
            w = (1 - beta_w)*w_free_inf + beta_w*fp_w

        else:
          print("LAST", T_post_cond, S_post_inf, sp["w_pre_max"], S_post_inf, S_post_free_inf)
          pass

    elif T_pre_free == T_post_free:
      print("Balanced homeostasis")

    else:
      if t < T_post_free:
        w = sp["lmbda"]*p["ij"][post_i, pre_j]*t
        tau_w = 0
        fp_w = None
        S_pre = S_pre_0 + sp["lmbda"]*sp["K_post"]*p["j"][pre_j]*t
        S_post = S_post_0 + sp["lmbda"]*sp["K_pre"]*p["i"][post_i]*t
      else:
        S_pre_free_inf = S_pre_0 + sp["lmbda"]*sp["K_post"]*p["j"][pre_j]*T_post_free
        S_pre_inf = sp["w_post_max"]*sp["K_pre"]*p["j"][pre_j]/(sp["K_post"]*p["i"][post_i])
        tau_pre = sp["w_post_max"]/(sp["lmbda"]*sp["K_post"]*p["i"][post_i])
        #S_post_inf = sp["w_pre_max"]*sp["K_post"]*p["i"][post_i]/(sp["K_pre"]*0.3)
        #tau_post = sp["w_pre_max"]/(sp["lmbda"]*sp["K_pre"]*0.3)

        T_pre_cond = -tau_pre*np.log((S_pre_inf - sp["w_post_max"])/(S_pre_inf - S_pre_free_inf)) if (S_pre_inf - sp["w_pre_max"] > 0) and (S_pre_free_inf != S_pre_inf) else np.inf
        w_free_inf = sp["lmbda"]*p["ij"][post_i, pre_j]*T_post_free
        tau_w = sp["w_post_max"]/(sp["lmbda"]*sp["K_pre"]*p["i"][post_i])
        fp_w = sp["w_post_max"]*p["ij"][post_i, pre_j]/(sp["K_pre"]*p["i"][post_i])
        if (t < T_pre_cond):
            S_post = sp["w_post_max"]
            beta_pre = 1 - np.exp(-(t - T_post_free)/tau_pre)
            S_pre = (1 - beta_pre)*S_pre_free_inf + beta_pre*S_pre_inf
            beta_w = 1 - np.exp(-(t - T_post_free)/tau_w)
            w = (1 - beta_w)*w_free_inf + beta_w*fp_w

        else:
          print("LAST", T_post_cond, S_post_inf, sp["w_pre_max"], S_post_inf, S_post_free_inf)
          pass

      pass

    return w, S_pre, S_post if only_vars else (T_pre_free, T_post_free, S_pre, S_post, tau_w, fp_w, w)
