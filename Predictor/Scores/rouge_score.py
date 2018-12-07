from Predictor.Scores.rouge import Rouge
import numpy as np
import ipdb


rouge = Rouge()

def rouge_func(predict, target):
    scores = []
    for (ps, ta) in zip(predict, target):
        ps = ps[ps.nonzero()]
        ta = ta[ta.nonzero()]
        score = rouge.calc_score([ps], [ta])
        scores.append(score)
    score = np.mean(scores)
    return score