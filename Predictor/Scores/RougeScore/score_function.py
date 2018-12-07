from .rouge import Rouge
import numpy as np


rouge = Rouge()


def rouge_func(predict, target):
    scores = []
    for (ps, ta) in zip(predict, target):
        score = rouge.calc_score([ps], [ta])
        scores.append(score)
    score = np.mean(scores)
    return score





