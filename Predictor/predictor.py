import torch as t
from Predictor import Models
from Predictor.vocab import load_vocab
import os
from configs import DefaultConfig


def get_best_k_model_path(path, k=1):
    k_best_model_folder = sorted(os.listdir(path), key=lambda x: x.split('T')[2], reverse=True)[:k]
    return k_best_model_folder


class Caption:
    def __init__(self, exp_root):
        self.vocab = load_vocab()
        self._load(exp_root)

    def _load(self, load_from_exp):
        best_model_path = get_best_k_model_path(load_from_exp, k=1)[0]
        trainner_state = t.load(os.path.join(load_from_exp, best_model_path, 'trainner_state'))
        self.args = trainner_state['args']
        self.model = getattr(Models, self.args.model_name)(self.vocab, self.args)
        self.model.load_state_dict(t.load(os.path.join(load_from_exp, best_model_path, 'model')))

    def predict(self, image_tensor):
        return self.model(image_tensor)

