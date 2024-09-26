import numpy as np

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
              print("hey 0")
              p["ij"][neuron_i, neuron_j] = 0
            if element_i == element_j:
              if attribute_i == 0:
                print("hey 1")
                p["ij"][neuron_i, neuron_j] = np.sum(p_ij_latent[element_i])
              else:
                print("hey 2")
                p["ij"][neuron_i, neuron_j] = np.sum(p_ij_latent[:, element_i])
          else:
            print("hey 3")
            (element_0, element_1) = (element_i, element_j) if attribute_i == 0 else (element_j, element_i)
            p["ij"][neuron_i, neuron_j] = p_ij_latent[element_0, element_1]
        
        p["i"] = network_sparsity*np.ones((input_size))
        p["j"] = network_sparsity*np.ones((input_size))

        p["j"][100:150] = 0.8
        p["j"][150:200] = 0.2

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