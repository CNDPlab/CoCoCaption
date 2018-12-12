from .base import BaseTrainerS
from tqdm import tqdm
import torch as t
import numpy as np
from pycocoevalcap.rouge.rouge import Rouge


rouge = Rouge()


class ScheduledTrainerTrans(BaseTrainerS):
    def __init__(self, args, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=True):
        super(ScheduledTrainerTrans, self).__init__(args, model, loss_func, score_func, train_loader, dev_loader, use_multi_gpu=use_multi_gpu)

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_epoch()
            self.global_epoch += 1
            if self.global_epoch == 15:
                try:
                    self.model.vgg_feature.requires_grad = True
                except:
                    self.model.module.vgg_feature.requires_grad = True
                print('vgg requires_grad set True.')
            self.reserve_topk_model(5)
        if self.summary_writer:
            self.summary_writer.close()
        print(f'Done')

    def train_epoch(self):
        for data in tqdm(self.train_loader, desc='train step'):
            train_loss = self.train_inference(data)
            train_loss.backward()
            t.nn.utils.clip_grad_norm_([i for i in self.model.parameters() if i.requires_grad is True], max_norm=5.0)
            self.optim.step()

            if self.summary_writer:
                self.summary_writer.add_scalar('loss/train_loss', train_loss.item(), self.global_step)
                self.summary_writer.add_scalar('lr', self.optim.state_dict()['param_groups'][0]['lr'], self.global_step)
            self.global_step += 1
        eval_score = self.evaluation()
        self.save(eval_score)
        self.scheduler.step(eval_score)

    def evaluation(self):
        scores = []
        self.model.eval()
        with t.no_grad():
            for data in tqdm(self.dev_loader, desc='eval_step'):
                score = self.eval_inference(data)
                scores.append(score)
        eval_score = np.mean(scores)
        if self.summary_writer:
            self.summary_writer.add_scalar('score/eval_score', eval_score, self.global_step)

        self.scheduler.step(eval_score)
        self.model.train()
        return eval_score

    def train_inference(self, data):
        self.model.train()
        feature, caption, lenths = [i.cuda() for i in data]
        batch_size, c, h, w = feature.size()
        _, n, l = caption.size()
        feature = feature.unsqueeze(1).expand((batch_size, n, c, h, w)).contiguous().view(-1, c, h, w)
        caption = caption.long()
        caption = caption.view(-1, l)

        self.optim.zero_grad()

        output_log_prob, output_id = self.model(feature, caption)
        loss = self.loss_func(output_log_prob, caption)
        return loss

    def eval_inference(self, data):
        feature, caption, lenths = [i.cuda() for i in data]
        caption = caption.long()
        if self.use_multi_gpu:
            output_id = self.model.module.greedy_search(feature)
        else:
            output_id = self.model.greedy_search(feature)

        scores = []
        cutted_output_id = self.cut_output_token(output_id.cpu().tolist())
        for i, v in enumerate(caption):
            cutted_caption = self.cut_output_token(v.cpu().tolist())
            score = rouge.calc_score([cutted_output_id[i]], cutted_caption)
            scores.append(score)
        score = np.mean(scores)
        return score

    def cut_output_token(self, output_id):
        cutted_output_id = []
        for instance in output_id:
            ninstance = []
            for id in instance:
                if self.use_multi_gpu:
                    if id != self.model.module.vocab.t2i['<EOS>']:
                        ninstance.append(id)
                    else:
                        ninstance.append(id)
                        break
                else:
                    if id != self.model.vocab.t2i['<EOS>']:
                        ninstance.append(id)
                    else:
                        ninstance.append(id)
                        break
            cutted_output_id.append(ninstance[1:])
        return cutted_output_id


