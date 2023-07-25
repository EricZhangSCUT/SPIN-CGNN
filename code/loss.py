import numpy as np

class MultiTermLossFunction(object):
    def __init__(self, loss_fn, weight) -> None:
        self.terms = list(weight.keys())
        if type(loss_fn) is not dict:
            self.loss_fn =  {term:loss_fn for term in self.terms}
        else:
            self.loss_fn = loss_fn
        self.weight = weight

    def __call__(self, inp):
        loss = {}
        for term in self.terms:
            pred, tgt = inp[term]
            loss[term] = self.loss_fn[term](pred, tgt)
        
        loss['Loss'] = sum([self.weight[term]*loss[term] for term in self.terms])
        return loss


class LossManager(object):
    def __init__(self, loss_fn) -> None:
        self.collecter = []
        if type(loss_fn) is MultiTermLossFunction:
            self.terms = loss_fn.terms
            self.multiloss_collecter = {term:[] for term in self.terms}
            self.append = self.append_multiloss
            self.log = self.log_multiloss
        else:
            self.append = self.append_loss
            self.log = self.log_loss
    
    def append_multiloss(self, loss):
        self.append_loss(loss['Loss'])
        for term in self.terms:
            self.multiloss_collecter[term].append(loss[term].item())
    
    def append_loss(self, loss):
        self.collecter.append(loss.item())

    def log_loss(self, latest=0):
        return 'Loss %.4f' % np.mean(self.collecter[-latest:])

    def log_multiloss(self, latest=0):
        loss_string = self.log_loss(latest)
        items_string = ' / '.join(['%s %.4f' % (term, np.mean(self.multiloss_collecter[term][-latest:])) for term in self.terms])
        return f'{loss_string} [{items_string}]'