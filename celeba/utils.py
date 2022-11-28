import os
import subprocess as sp
import numpy as np
import torch


def check_gpu():
    available_gpu = -1
    ACCEPTABLE_USED_MEMORY = 1000
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    memory_used_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_used_values = [int(x.split()[0]) for i, x in enumerate(memory_used_info)]
    for k in range(len(memory_used_values)):
        if memory_used_values[k]<ACCEPTABLE_USED_MEMORY:
            available_gpu = k
            break
    return available_gpu



def alignments(grads):
    """
    Computes the alignment of tasks' gradients to the overall filters's gradients.
    """
    tasks_nb, filters_nb, depth, w, h = grads.shape
    # Reshape to flatten the filters
    flat_grads = np.reshape(grads, (tasks_nb, w*h*depth, filters_nb))
    # Total gradients per filter (sum over tasks)
    total_filters_grads = np.sum(flat_grads, axis=0)
    # Compute scalar products between each task/filter gradients and total gradients per filter
    norms = np.linalg.norm(total_filters_grads, axis=0)*np.linalg.norm(flat_grads, axis=1) # norms of both terms
    grads_filters_alignment = np.sum(flat_grads*total_filters_grads, axis=1) # sum of elementwise products
    grads_filters_alignment[norms>0] /= norms[norms>0] # division, only for non-nul elements
    return grads_filters_alignment


def get_loss_rate(task_losses, loss_prev):
    n = len(task_losses)
    print("N = ", n)
    rate = [0]*n
    if len(loss_prev) < len(task_losses):
        b = [1] * (len(loss_prev) - n)
        b_tensor = torch.tensor(b)
        loss_prev = torch.cat(loss_prev, b_tensor)
    for i in range(n):
        rate[i] = task_losses[i]/loss_prev[i]
    max_rate = rate[0]
    index = 0
    for i in range(n):
        if max_rate < rate[i]:
            max_rate = rate[i]
            index = i
    return rate, index        