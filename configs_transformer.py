import warnings


class DefaultConfig:
    model_name = 'VGGTransformerNew1'
    ckpt_root = 'Predictor/checkpoints'

    dropout = 0.1
    num_head = 6
    embedding_size = 512
    max_seq_len = 65
    epochs = 30
    batch_size = 10
    hidden_size = 128
    eval_every_step = 5
    save_every_step = 5
    resume = None

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print('__', k, getattr(self, k))

