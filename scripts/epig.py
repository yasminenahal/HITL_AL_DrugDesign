import torch

import logging
import math

# functions from Freddie's code needed to compute the epig scores
def conditional_epig_from_probs(
    probs_pool: torch.Tensor, 
    probs_targ: torch.Tensor
    ) -> torch.Tensor:
    """
    See conditional_epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the joint predictive distribution.
    probs_pool = probs_pool.permute(1, 0, 2)  # [K, N_p, Cl]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_pool = probs_pool[:, :, None, :, None]  # [K, N_p, 1, Cl, 1]
    probs_targ = probs_targ[:, None, :, None, :]  # [K, 1, N_t, 1, Cl]
    probs_pool_targ_joint = probs_pool * probs_targ
    probs_pool_targ_joint = torch.mean(probs_pool_targ_joint, dim=0)

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=0)
    probs_targ = torch.mean(probs_targ, dim=0)

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_pool_targ_joint and probs_pool_targ_joint_indep.
    nonzero_joint = probs_pool_targ_joint > 0
    log_term = torch.clone(probs_pool_targ_joint)
    log_term[nonzero_joint] = torch.log(probs_pool_targ_joint[nonzero_joint])
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])
    scores = torch.sum(probs_pool_targ_joint * log_term, dim=(-2, -1))
    return scores  # [N_p, N_t]

def conditional_mse_from_predictions(predictions_pool: torch.Tensor, predictions_targ: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean squared error (MSE) between the predicted values for pairs of examples.
    Suitable for regression models.

    Arguments:
        predictions_pool: Tensor[float], [N_p]
        predictions_targ: Tensor[float], [N_t]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    predictions_pool = predictions_pool.unsqueeze(1)  # [N_p, 1]
    predictions_targ = predictions_targ.unsqueeze(0)  # [1, N_t]
    mse = torch.mean((predictions_pool - predictions_targ) ** 2, dim=-1)  # [N_p, N_t]
    return mse

def check(
    scores: torch.Tensor, max_value: float = math.inf, epsilon: float = 1e-6, score_type: str = ""
) -> torch.Tensor:
    """
    Warn if any element of scores is negative, a nan or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores + epsilon >= 0) & (scores - epsilon <= max_value)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        
        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")
    
    return scores

def epig_from_conditional_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        scores: Tensor[float], [N_p, N_t]

    Returns:
        Tensor[float], [N_p,]
    """
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, score_type="EPIG")  # [N_p,]
    return scores  # [N_p,]

def epig_from_probs(probs_pool: torch.Tensor, probs_targ: torch.Tensor, classification: str = True) -> torch.Tensor:
    """
    See epig_from_logprobs.

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    if classification:
        scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    else:
        scores = conditional_mse_from_predictions(probs_pool, probs_targ)
    return epig_from_conditional_scores(scores)  # [N_p,]
