import os
import numpy as np
import pandas as pd
import json
from copy import deepcopy
import re
import random
from random import choice
random.seed(10)
# -*- coding:utf-8 -*

def sm_lr(file, label, convar, disvar):

    file.columns = [i.replace(" ", "_") for i in file.columns]
    convar = [i.replace(" ", "_") for i in convar]
    disvar = [i.replace(" ", "_") for i in disvar]
    convar = "+".join(convar)
    disvar = "+".join(disvar)
    
    import statsmodels.formula.api as smf
    f1 = label + " ~ " + convar + disvar
    model = smf.logit(f1, data = file)
    glm1 = model.fit(method='bfgs')
    print(glm1.aic)
    print(glm1.summary())

    # Note that tables is a list. The table at index 1 is the "core" table. Additionally, read_html puts dfs in a list, so we want index 0
    results_summary = glm1.summary()
    results_summary.tables[1]
    return glm1, results_summary.tables[1]


def auc(label_l, prepro_l, fig_path):
    
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, thersholds = roc_curve(label_l, prepro_l)
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)
    
    fig  = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig("%s/roc_curve.png" % (fig_path))

    return thersholds[(tpr-fpr).tolist().index(max(tpr-fpr))]

def evaluate(label_l, prepro_l, cut_value, posi_label):

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score 
    standard = pd.DataFrame(label_l)
    prediction = pd.DataFrame(prepro_l)
    prediction.loc[prediction[0]>=cut_value] = 1
    prediction.loc[prediction[0]<cut_value] = 0

    accuracy = accuracy_score(standard[0],
                            prediction[0],
                            normalize=True,
                            sample_weight=None)
    precision = precision_score(standard[0],
                                prediction[0],
                                labels=None,
                                pos_label=posi_label,
                                average='binary',
                                sample_weight=None,
                                zero_division='warn')
    recall = recall_score(standard[0],
                        prediction[0],
                        labels=None,
                        pos_label=posi_label,
                        average='binary',
                        sample_weight=None,
                        zero_division='warn')
    f1 = f1_score(standard[0],
                prediction[0],
                labels=None,
                pos_label=posi_label,
                average='binary',
                sample_weight=None,
                zero_division='warn')

    print("结果:", accuracy, precision, recall, f1)
    
    return accuracy, precision, recall, f1