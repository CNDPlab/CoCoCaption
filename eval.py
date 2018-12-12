import torch as t
from Predictor import Models
from vocabulary import Vocab
import os
from PIL import Image
from configs_transformer import DefaultConfig
from Predictor.Utils.picture_transform import picture_tranform_func


ckpt_root = DefaultConfig.ckpt_root


def get_best_k_model_path(path, k=1):
    k_best_model_folder = sorted(os.listdir(path), key=lambda x: x.split('T')[1], reverse=True)[:k]
    return k_best_model_folder


def load_model(exp_name):
    exp_root = os.path.join(ckpt_root, exp_name)
    best_model_folder = get_best_k_model_path(os.path.join(exp_root, 'saved_models'))[0]
    best_model_folder = os.path.join(exp_root, 'saved_models', best_model_folder)
    model_state = t.load(os.path.join(best_model_folder, 'model'))
    for i in model_state:
        model_state[i] = model_state[i].cpu()
    trainner_state = t.load(os.path.join(best_model_folder, 'trainner_state'))
    args = trainner_state['args']

    vocab = Vocab()
    model = getattr(Models, args.model_name)(vocab, args)
    model.load_state_dict(model_state)
    model.eval()
    return model


class Inference:
    def __init__(self, exp_name='20181211_203902'):
        self.vocab = Vocab()
        self.model = load_model(exp_name)
        self.model.eval()

    def predict(self, picture_path):
        with t.no_grad():
            image = Image.open(picture_path)
            image = picture_tranform_func(image)
            image = image.unsqueeze(0)

            output = self.model.greedy_search(image)
            output = output.tolist()
            output = self.model.vocab.convert(output,'li2t')
        return output

    def predict_many(self, picture_root):
        with t.no_grad():
            pictures = os.listdir(picture_root)
            images = [Image.open(os.path.join(picture_root, i)) for i in pictures]
            images = [picture_tranform_func(i) for i in images]
            images = t.stack(images, 0)
            output = self.model.greedy_search(images)
            output = output.tolist()
            output = self.model.vocab.convert(output, 'li2t')
        return output

inference = Inference()
inference.predict_many('image_folder')
