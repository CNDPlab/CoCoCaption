import fire
from configs_transformer import DefaultConfig
import pickle as pk
from Predictor import Models
from Predictor.Scores import rouge_func
from Predictor.Losses import transformer_celoss
from Predictor.predictor import Caption
from loaders import get_loaders
from Trainer import ScheduledTrainerTrans


def train(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    vocab = pk.load(open(args.vocab, 'rb'))
    loss_functions = transformer_celoss
    score_functions = rouge_func
    model = getattr(Models, args.model_name)(vocab, args)
    train_loader = get_loaders('train', args.batch_size)
    dev_loader = get_loaders('val', args.batch_size)
    trainer = ScheduledTrainerTrans(args, model, loss_functions, score_functions, train_loader, dev_loader)
    if args.resume is not None:
        trainer.init_trainner(resume_from=args.resume)
    else:
        trainer.init_trainner()
    trainer.train()

def predict():
    exp_root = None
    caption = Caption(exp_root=exp_root)
    caption.predict()



if __name__ == '__main__':
    fire.Fire()
