"""
Author: Subhabrata Mukherjee (submukhe@microsoft.com)
Code for Uncertainty-aware Self-training (UST) for few-shot learning.
"""

from sklearn.utils import shuffle

import logging
import numpy as np
import os
import random


logger = logging.getLogger('UST')

def get_BALD_acquisition(y_T):

	expected_entropy = - np.mean(np.sum(y_T * np.log(y_T + 1e-10), axis=-1), axis=0) 
	expected_p = np.mean(y_T, axis=0)
	entropy_expected_p = - np.sum(expected_p * np.log(expected_p + 1e-10), axis=-1)
	return (entropy_expected_p - expected_entropy)

def sample_by_bald_difficulty(X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by difficulty BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), BALD_acq)
	p_norm = p_norm / np.sum(p_norm)
	indices = np.random.choice(len(X), num_samples, p=p_norm, replace=False)
	X_s = [X[i] for i in indices]
	y_m = y_mean[indices]
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_m, w_s, y_s, BALD_acq, p_norm


def sample_by_bald_easiness(X, y_mean, y_var, y, num_samples, num_classes, y_T):

	logger.info ("Sampling by easy BALD acquisition function")
	BALD_acq = get_BALD_acquisition(y_T)
	p_norm = np.maximum(np.zeros(len(BALD_acq)), (1. - BALD_acq)/np.sum(1. - BALD_acq))
	p_norm = p_norm / np.sum(p_norm)
	# print("len(X)=", len(X))
	# print("num_samples=", num_samples)
	# print("p_norm=", p_norm)
	indices = np.random.choice(len(X), num_samples, p=p_norm, replace=False)
	X_s = [X[i] for i in indices]
	y_m = y_mean[indices]
	y_s = y[indices]
	w_s = y_var[indices][:,0]
	return X_s, y_m, w_s, y_s, BALD_acq, p_norm


# def sample_by_bald_class_easiness(X, y_mean, y_var, y, num_samples, num_classes, y_T):

# 	logger.info ("Sampling by easy BALD acquisition function per class")
# 	BALD_acq = get_BALD_acquisition(y_T)
# 	BALD_acq = (1. - BALD_acq)/np.sum(1. - BALD_acq)
# 	logger.info (BALD_acq)
# 	samples_per_class = num_samples // num_classes
# 	X_s_input_ids, X_s_attention_mask, y_s, w_s = [], [], [], []
# 	for label in range(num_classes):
# 		X_input_ids, X_attention_mask = X['input_ids'][y == label], X['attention_mask'][y == label]
# 		y_ = y[y==label]
# 		y_var_ = y_var[y == label]		
# 		# p = y_mean[y == label]
# 		p_norm = BALD_acq[y==label]
# 		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
# 		p_norm = p_norm/np.sum(p_norm)
# 		if len(X_input_ids) < samples_per_class:
# 			logger.info ("Sampling with replacement.")
# 			replace = True
# 		else:
# 			replace = False
# 		indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
# 		X_s_input_ids.extend(X_input_ids[indices])
# 		X_s_attention_mask.extend(X_attention_mask[indices])
# 		y_s.extend(y_[indices])
# 		w_s.extend(y_var_[indices][:,0])
# 	X_s_input_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_attention_mask, y_s, w_s)
# 	return {'input_ids': np.array(X_s_input_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)


# def sample_by_bald_class_difficulty( X, y_mean, y_var, y, num_samples, num_classes, y_T):

# 	logger.info ("Sampling by difficulty BALD acquisition function per class")
# 	BALD_acq = get_BALD_acquisition(y_T)
# 	samples_per_class = num_samples // num_classes
# 	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = [], [], [], [], []
# 	for label in range(num_classes):
# 		X_input_ids, X_token_type_ids, X_attention_mask = X['input_ids'][y == label], X['token_type_ids'][y == label], X['attention_mask'][y == label]
# 		y_ = y[y==label]
# 		y_var_ = y_var[y == label]		
# 		p_norm = BALD_acq[y==label]
# 		p_norm = np.maximum(np.zeros(len(p_norm)), p_norm)
# 		p_norm = p_norm/np.sum(p_norm)
# 		if len(X_input_ids) < samples_per_class:
# 			replace = True
# 			logger.info ("Sampling with replacement.")
# 		else:
# 			replace = False
# 		indices = np.random.choice(len(X_input_ids), samples_per_class, p=p_norm, replace=replace)
# 		X_s_input_ids.extend(X_input_ids[indices])
# 		X_s_token_type_ids.extend(X_token_type_ids[indices])
# 		X_s_attention_mask.extend(X_attention_mask[indices])
# 		y_s.extend(y_[indices])
# 		w_s.extend(y_var_[indices][:,0])
# 	X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s = shuffle(X_s_input_ids, X_s_token_type_ids, X_s_attention_mask, y_s, w_s)
# 	return {'input_ids': np.array(X_s_input_ids), 'token_type_ids': np.array(X_s_token_type_ids), 'attention_mask': np.array(X_s_attention_mask)}, np.array(y_s), np.array(w_s)
