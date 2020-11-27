from __future__ import annotations
import numpy as np
from typing import List, Dict

def ndcg_score(y_true,
               y_pred, 
               item_data: List[Dict],
               n_predictions: int = 10, 
               item_hit_relevance: int = 15, 
               domain_hit_relevance: int = 1):
    
    """
    Returns ndcg score for submission ('y_pred') with regard to a target
    list of items bought ('y_true'). To do so, it uses the items metadata
    provided in item_data.jl.gz file

    :param y_true: list containing target item-ids.
    :param y_pred: list of recommendation sets for each item bought in y_true.
    :param item_meta: list of items metadata as in item_data.jl.gz file.
    :param n_predictions: required length of each recommendation set.
    :param item_hit_relevance: yield for matching the target item-id.
    :param domain_hit_relevance: yield for matching the target domain-id.

    """
    #preprocess inputs
    y_true, y_pred, y_true_domain, y_pred_domain = _preprocess_inputs(y_true, 
                                                                      y_pred, 
                                                                      item_data,
                                                                      n_predictions)
    
    #compute submission relevance
    submission_relevance = _get_submission_relevance(y_true, y_pred,
                                                     y_true_domain, y_pred_domain,
                                                     item_hit_relevance,
                                                     domain_hit_relevance)
    #compute ideal relevance
    ideal_relevance = np.array([item_hit_relevance] +
                               [domain_hit_relevance] * (n_predictions - 1))[None]
                               
    #return metric
    return _ndcg(submission_relevance, ideal_relevance, n_predictions)


#Auxiliary functions

def _preprocess_inputs(y_true, y_pred, item_data,n_predictions):
    
    """
    Preprocess the score function inputs and generates the corresponding 
    domain-vectors needed for scoring. 
    
    :param y_true: list containing target item-ids.
    :param y_pred: list of recommendation sets for each item bought in y_true
    :param item_meta: list of items metadata as in item_data.jl.gz file
    :param n_predictions: required length of each recommendation set 
    
    """
    
    y_true = np.array(y_true, dtype=np.int)
    y_pred = _preprocess_submission(y_pred,n_predictions)
    
    item_meta = {i['item_id']:i for i in item_data}
    item2domain = {k:v['domain_id'] for k,v in item_meta.items()}
    i2d = np.vectorize(lambda x: item2domain[x] if x else -1)
    
    y_pred_domain = i2d(y_pred)
    y_true_domain = i2d(y_true)
    
    
    return (y_true, y_pred, y_true_domain, y_pred_domain)
    
    
def _preprocess_submission(y_pred,n_predictions):
    
    """
    Preprocess the recommended items to make sure that: 
    a. there are no duplicated items within a recommendation set
    b. all recommendations set have len n_predictions (fill with 'None').
    
    :param y_pred: list of recommendation sets.
    :param n_predictions: required length of each recommendation set.
    
    """
    
    processed_rows = []
    for row in y_pred:
        # remove duplicates and keep order
        set_ = set()
        for i, x in enumerate(row):
            if x not in set_ and x:
                row[i] = int(x)
                set_.add(x)
            else:
                row[i] = None
       
        # pad row if missing predictions
        row = row + [None] * (n_predictions - len(row))
        processed_rows.append(row)
        
    return np.array(processed_rows)
    
def _get_submission_relevance(y_true, y_pred, y_true_domain,y_pred_domain, 
                              item_hit_relevance,domain_hit_relevance):
    
    """
    Given the target and the predicted list of items and domains 
    from a submission, computes relevance with the specified parameters.
    
    
    :param y_true: array of ints, corresponding to target item-ids.
    :param y_pred: array of lists, corresponding to recommended item-ids.
    :param y_true_domain: array of ints, corresponding to target domain-ids.
    :param y_pred_domain: array of lists, corresponding to recommended domain-ids.
    :param item_hit_relevance: yield for matching the target item-id.
    :param domain_hit_relevance: yield for matching the target domain-id.

    """
        
    # item relavance
    rel_item = np.zeros_like(y_pred)
    idx = np.argwhere(np.equal(y_pred, y_true[:,None]))
    if len(idx):
        x_idx, y_idx = idx[:, 0], idx[:, 1]
        rel_item[x_idx, y_idx] = item_hit_relevance
        
    # domain relevance
    rel_domain = (np.char.equal(y_pred_domain, y_true_domain[:,None])
                 ) * domain_hit_relevance
                
    return np.maximum(rel_item, rel_domain)

def _dcg(relevance,n_predictions,mean=True):
    
    """
    Computes the discounted cumulative gain for a given relevance, considering
    a certain number of recommendations in each set, then takes 
    the average across rows. 
    
    :param relevance: array of recommendations relevances.
    :param n_predictions: length of each recommendation set.
    :mean: whether or not dcg show be averaged across rows (default True).
    
    """
    
    x = np.arange(n_predictions) + 1
    dcg = np.sum(relevance / np.log2(x + 1), axis=1)
    dcg = np.mean(dcg) if mean else dcg
    return dcg

def _ndcg(submission_relevance, ideal_relevance,n_predictions):
    
    """
    Given the relevance of a submission and its corresponding ideal relevance.
    returns the average normalized cumulative discounted gain.
    
    :param submission_relevance: array of recommendation relevances for a certain submission
    (as an output from _get_submission_relevance function).
    :param ideal_relevance: array of relevances of the ideal recommendation
    :param n_predictions: length of each recommendation set.
    
    """
    
    return (_dcg(submission_relevance,n_predictions)/
            _dcg(ideal_relevance,n_predictions))
    
    
    
    
    
    
    