import argparse
import numpy as np

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from statistic_utils import process_npy, data_with_metrics

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--exp', help='experiment name')


def dataset_to_length_and_batch_size(dataset, task=None):
    if dataset == 'cifar10':
        return 50000, None


# args = parser.parse_args()


def get_robust_ind(your_path, percentage):
    len_dataset,_ = dataset_to_length_and_batch_size("cifar10", None)
    # your_path = f"../{args.exp}.npy"
    new_data_loss_diff, new_data_original_correctness, new_data_flip_times, \
    new_data_delta_grad, new_data_original_loss, new_data_perturbed_loss, new_data_original_logit, \
    new_data_perturbed_logit, new_data_logit_diff, new_data_original_probability, \
    new_data_perturbed_probability, new_data_probability_diff,new_data_golden_label= process_npy(
        your_path, # todo set your statistic path here
        len_dataset=len_dataset,use_normed_loss=False,use_delta_grad=False,
        only_original_pred=False
    )

    df = data_with_metrics( new_data_loss_diff,
                            new_data_original_correctness,
                            new_data_flip_times,
                            new_data_delta_grad,
                            new_data_original_loss,
                            new_data_perturbed_loss,
                            new_data_original_logit=new_data_original_logit,
                            new_data_perturbed_logit=new_data_perturbed_logit,
                            new_data_logit_diff=new_data_logit_diff,
                            new_data_original_probability=new_data_original_probability,
                            new_data_perturbed_probability=new_data_perturbed_probability,
                            new_data_probability_diff=new_data_probability_diff,
                            new_data_golden_label=new_data_golden_label,
                            do_norm=True)

    # sorted(sorted(new_data_perturbed_loss))
    n = int(percentage * len_dataset)
    sorted_df = df.sort_values(['perturbed_loss_mean', 'perturbed_loss_std', 'flip_times'], ascending=[True, True, True])
    most_robust = sorted_df.iloc[0:n, :]
    least_robust = sorted_df.iloc[-n:, :]
    swing = sorted_df.iloc[25000 - n//2:25000 + n//2, :]
    return most_robust['id'].tolist(), swing['id'].tolist(), least_robust['id'].tolist()