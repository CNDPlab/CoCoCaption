import fire
from configs_transformer import DefaultConfig
from vocabulary import Vocab
from Predictor import Models
from Predictor.Scores import rouge_func
from Predictor.Losses import transformer_celoss
from loaders import get_loaders
from Trainer import ScheduledTrainerTrans


def train(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    vocab = Vocab()
    loss_functions = transformer_celoss
    score_functions = rouge_func
    model = getattr(Models, args.model_name)(vocab, args)
    train_loader = get_loaders('train', args.batch_size, 10)
    dev_loader = get_loaders('val', args.batch_size, 10)
    trainer = ScheduledTrainerTrans(args, model, loss_functions, score_functions, train_loader, dev_loader)
    if args.resume is not None:
        trainer.init_trainner(resume_from=args.resume)
    else:
        trainer.init_trainner()
    trainer.train()



def train_re(**kwargs):
    args = DefaultConfig()
    args.parse(kwargs)
    vocab = Vocab()
    loss_functions = transformer_celoss
    score_functions = rouge_func
    model = getattr(Models, args.model_name)(vocab, args)
    train_loader = get_loaders('train', args.batch_size, 10)
    dev_loader = get_loaders('val', args.batch_size, 10)
    trainer = ScheduledTrainerTrans(args, model, loss_functions, score_functions, train_loader, dev_loader)
    trainer.init_trainner(resume_from=args.resume)

    trainer.train()


if __name__ == '__main__':
    fire.Fire()
