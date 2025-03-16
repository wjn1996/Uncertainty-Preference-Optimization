import os
import json
import torch
import numpy as np
from tqdm import tqdm
import random
from random import shuffle

def mc_estimation(
    model,
    preference_pair_examples = None,
    preference_pair_num: int = -1,
    description: str = "Evaluate on Unlabeled Data via MC Dropout Uncertainty Estimation",
    prediction_loss_only = None,
    T: int = 30,
    num_classes: int = 2
):
    """
    Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

    MC dropout for esimating the pseudo-labeled preference pair data from reward model.
    """
    is_sample = True
    if preference_pair_num == -1 or preference_pair_num >= len(preference_pair_examples):
        preference_pair_num = len(preference_pair_examples)
        is_sample = False

    model.train() # 开启train模式，允许模型进行Dropout

    y_T = list()

    print("starting estimating ...")
    for i in range(T):
        with torch.no_grad():
            print(f"dropout iter {i + 1}")
            y_pred = []
            for step, inputs in enumerate(tqdm(preference_pair_examples)):
                # inputs = {k: v.to(model.device)for k, v in inputs.items()}
                outputs = model(**inputs, return_dict=True)
                logits = outputs.logits
                y_pred.extend(logits.detach().cpu().numpy().tolist())
            # print("y_pred.shape=", torch.Tensor(y_pred).shape) # [n, num_class]
            predict_proba = torch.softmax(torch.Tensor(y_pred).to(logits.device), -1)
            # print("predict_proba.shape=", predict_proba.shape) # [n, num_class]
            # y_T[i] = predict_proba.detach().cpu().numpy().tolist()
            y_T.append(predict_proba.detach().cpu().numpy().tolist())
    
    y_T = np.array(y_T) # [T, preference_pair_num, n_class]
    #compute mean
    y_mean = np.mean(y_T, axis=0)
    # print("y_mean.shape=", y_mean.shape) # e.g., (4095, 3) [n, class_num]
    # print("(preference_pair_num, num_classes)=", (preference_pair_num, num_classes))
    assert y_mean.shape == (preference_pair_num, num_classes)

    #compute majority prediction
    y_pred = np.array([np.argmax(np.bincount(row)) for row in np.transpose(np.argmax(y_T, axis=-1))])
    assert y_pred.shape == (preference_pair_num,)

    #compute variance
    y_var = np.var(y_T, axis=0)
    assert y_var.shape == (preference_pair_num, num_classes)

    """
    output example:
        y_mean[i]= [0.92937551 0.0706245 ]
        y_var[i]= [5.01353434e-08 5.01321431e-08]
        y_pred[i]= 0
        y_T[i]= [[9.99999166e-01 7.96695588e-07]
        [9.99497294e-01 5.02729556e-04]
        [9.99999762e-01 2.41698444e-07]
        [2.44081928e-08 1.00000000e+00]
        [9.99953270e-01 4.67648424e-05]
        [9.99991894e-01 8.14051600e-06]
        [1.06898369e-03 9.98930991e-01]
        [9.99997497e-01 2.53872940e-06]
        [1.33190450e-04 9.99866843e-01]
        [9.29519951e-01 7.04800189e-02]]
    """
    # for i in range(10):
    #     print("======")
    #     print("y_mean[i]=", y_mean[i])
    #     print("y_var[i]=", y_var[i])
    #     print("y_pred[i]=", y_pred[i])
    #     print("y_T[i]=", y_T[i])


    return y_mean, y_var, y_pred, y_T