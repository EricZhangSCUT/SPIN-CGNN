import os
import time
import argparse
import shutil
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from utils import setup_exp, get_lcr
from loss import MultiTermLossFunction, LossManager
from data_preprocess import aa_encode, aa_decode
from cath_data import CATH_CGraph
from model import SPIN_CGNN


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cath')
parser.add_argument('-a', '--arch', type=str, default='CGNN')
parser.add_argument('-d', '--dim', type=int, default=128)
parser.add_argument('--heads', type=int, default=4)
parser.add_argument('-c', '--cutoff', type=int, default=10)
parser.add_argument('-v', '--virtual_atom', type=int, default=3)
parser.add_argument('-l', '--layers', type=int, default=10)
parser.add_argument('-b', '--batchsize', type=int, default=4096)
parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-n', '--noise', type=float, default=0.)

parser.add_argument('--no_amp', action='store_true', default=False)
parser.add_argument('--note', type=str, default='')


class Trainer(object):
    def __init__(self, model, train_dataset, val_dataset, args, logger):
        super().__init__()
        self.args = args
        self.model = model
        self.logger = logger
        
        self.amp = False if args.no_amp else True
        if self.amp:
            self.scaler = GradScaler()

        self.train_dataset = train_dataset
        self.val_data = DataLoader(dataset=val_dataset, pin_memory=True, num_workers=8)

        self.lr = args.lr * (args.batchsize / 1024)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr,
            steps_per_epoch=len(train_dataset)+1, epochs=args.epochs
        )

        self.loss_fn = MultiTermLossFunction(
            loss_fn={
                'Base': torch.nn.CrossEntropyLoss(),
            },
            weight={
                'Base': 1,
            }
        )

        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []

    def repack_train_data(self):
        '''repack to shuffle entrys in groups'''
        self.train_dataset.pack_entrys(self.args.batchsize)
        self.train_data = DataLoader(
            dataset=self.train_dataset, shuffle=True, pin_memory=True, num_workers=8)

    def preprocess(self, sample):
        nn_input = {item: sample[item][0].cuda(non_blocking=True) for item in [
            'coo', 'rel_pos', 'consistancy', 'edges', 'graph_id', 'soe'
        ]}
        target = {'seq': sample['seq'][0].cuda(non_blocking=True)}
        return nn_input, target

    def criteria(self, nn_output, target):
        loss = self.loss_fn({
            'Base': (nn_output['logits'], target['seq']),
        })
        return loss

    def train_epoch(self, print_freq=100):
        self.repack_train_data()

        self.logger.info('Epoch %s Training' % self.cur_epoch)
        start_time = time.time()
        self.model.train()
        cur_iter = 0
        n_iters = len(self.train_data)
        train_loss = LossManager(self.loss_fn)

        for sample in self.train_data:
            self.optimizer.zero_grad()
            cur_iter += 1
            nn_input, target = self.preprocess(sample)

            if self.amp:
                with autocast():
                    nn_output = self.model(noise=self.args.noise, **nn_input)
                    loss = self.criteria(nn_output, target)
            else:
                nn_output = self.model(noise=self.args.noise, **nn_input)
                loss = self.criteria(nn_output, target)
            
            train_loss.append(loss)

            if cur_iter % print_freq == 0:
                self.logger.info(
                    'Iters %d/%d Train%s' % (cur_iter, n_iters, train_loss.log(latest=print_freq)))

            if self.amp:
                self.scaler.scale(loss['Loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss['Loss'].backward()
                self.optimizer.step()

            self.lr_scheduler.step()

        self.train_loss.append(np.mean(train_loss.collecter))
        self.logger.info('Train%s' % train_loss.log())
        end_time = time.time()
        self.logger.info('TrainTime: %d sec\n' % (end_time - start_time))

    def val_epoch(self):
        self.logger.info('Epoch %s Validating' % self.cur_epoch)
        start_time = time.time()
        self.model.eval()
        loss = LossManager(self.loss_fn)

        with torch.no_grad():
            for sample in self.val_data:
                nn_input, target = self.preprocess(sample)
                nn_output = self.model(**nn_input)
                loss.append(self.criteria(nn_output, target))

            self.val_loss.append(np.mean(loss.collecter))
            self.logger.info('Val%s' % loss.log())

        end_time = time.time()
        self.logger.info('ValTime: %d sec\n' % (end_time - start_time))

class Test(object):
    def __init__(self, model, data, logger, output_path, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = DataLoader(dataset=data, pin_memory=True, num_workers=8)
        self.logger = logger
        self.output_path = output_path
        self.logits_path = f'{output_path}/logits'
        pathlib.Path(self.logits_path).mkdir(parents=True, exist_ok=True)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        
    def preprocess(self, sample):
        nn_input = {item: sample[item][0].cuda(non_blocking=True) for item in [
            'coo', 'rel_pos', 'consistancy', 'edges', 'graph_id', 'soe'
        ]}
        target = {'seq': sample['seq'][0].cuda(non_blocking=True)}
        return nn_input, target
    
    def log_recovery(self, recoverys, loss):
        self.logger.info('%.4f %.4f %.4f %.4f %.2f' %(
            float(recoverys.median().cpu()),
            float(recoverys.mean().cpu()),
            float(recoverys.max().cpu()),
            float(recoverys.min().cpu()),
            torch.exp(torch.cat(loss).mean())
            ))
    
    def test(self):
        start_time = time.time()
        self.model.eval()
        
        recoverys = []
        loss = [] 
        writer = open(f'{self.output_path}/pred_seq.fa', 'w')
        with torch.no_grad():
            for sample in self.data:
                name = sample['name'][0]
                nn_input, target = self.preprocess(sample)
                nn_output = self.model(**nn_input)
                logits = nn_output['logits']
                pred_S = nn_output['logits'].argmax(-1)
                
                l = self.loss_fn(nn_output['logits'], target['seq']).cpu()
                rec = (pred_S==target['seq']).float().mean()
                recoverys.append(rec)
                loss.append(l)

                np.save(f'{self.logits_path}/{name}.npy', logits.cpu().numpy())
                pred_seq = aa_decode(pred_S.cpu().numpy())
                writer.write(f'>{name}\n{pred_seq}\n')
        
        writer.close()
        self.log_recovery(torch.stack(recoverys), loss)
        lcr = get_lcr(path=self.output_path)*100
        self.logger.info('LCR %.2f'%lcr)
        
        end_time = time.time()
        self.logger.info('TestTime: %d sec\n' % (end_time - start_time))
        

def main():
    args = parser.parse_args()
    exp_name, save_dir, logger = setup_exp(args)

    train_data = CATH_CGraph(dataset=args.data, subset='train', batch_size=args.batchsize, cutoff=args.cutoff)
    val_data = CATH_CGraph(dataset=args.data, subset='validation', batch_size=args.batchsize, cutoff=args.cutoff)
    test_data = CATH_CGraph(dataset=args.data, subset='test', cutoff=args.cutoff)
    
    logger.info('Dataset:' + args.data)
    logger.info('Trainset : %s \t Valset: %s' %
                (len(train_data), len(val_data)))
    
    model = SPIN_CGNN(
        d_model=args.dim,
        n_heads=args.heads,
        n_layers=args.layers,
        n_virtual_atoms=args.virtual_atom,
    ).cuda()
    logger.info(model)
    logger.info('Number of Model Parameters: %.2f M' %
                (sum(p.numel() for p in model.parameters())/1e6))

    trainer = Trainer(model, train_data, val_data, args, logger)
    while trainer.cur_epoch < args.epochs:
        trainer.train_epoch()

        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        trainer.val_epoch()
        trainer.cur_epoch += 1

        if trainer.val_loss[-1] == min(trainer.val_loss):
            torch.save(model.state_dict(), os.path.join(
                save_dir, 'best_model.pth'))
    
    state_dict = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(state_dict)
    
    output_path = f'../output/{exp_name}/cath'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    test_runner_cath = Test(model, test_data, logger, output_path, args)
    test_runner_cath.test()

    shutil.copyfile('../log/%s.txt' %
                    exp_name, '../experiment/%s/log.txt' % exp_name)


if __name__ == '__main__':
    main()
