import os
import math
from decimal import Decimal

from utils import utility

import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from tqdm import tqdm

from utils.at_utils import *
import torch.nn.functional as F

class Trainer():
    def __init__(self, args, loader, model_s, model_t, loss, ckp):
        self.writer = SummaryWriter('experiment/RCAN_BIX2_G10R20P48')

        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test

        self.model_s = model_s      # student
        self.model_t = model_t      # teacher
        self.loss = loss
        self.optimizer = utility.make_optimizer(args, self.model_s)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch

        # epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        # start train
        self.loss.start_log()
        self.model_s.train()    

        timer_data, timer_model = utility.timer(), utility.timer()
        print(len(self.loader_train))
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            idx_scale = torch.IntTensor(0).to('cuda:0')
            lr, hr = self.prepare([lr, hr])
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            _, fs_t = self.model_t(lr)    # 教师网络返回中间层特征
            sr, fms_s = self.model_s(lr, idx_scale)    # 学生网络返回注意力map
            loss = self.loss(sr, hr, fms_s, fs_t)

            # 
            self.writer.add_scalar('Train/loss', loss.to('cpu').item(), batch)

            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        # epoch = self.scheduler.last_epoch + 1
        epoch = self.scheduler.last_epoch

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model_s.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare([lr, hr])
                    else:
                        lr = self.prepare([lr])[0]

                    if self.args.chop:
                        sr = self.model_s(lr, idx_scale)        #
                    else:
                        sr, _ = self.model_s(lr, idx_scale)        #

                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            # self.ckp.save(self, epoch, is_best=(best[1][0] == epoch))
            self.ckp.save(self, epoch, is_best=(best[1][0]+1 == epoch))


    def prepare(self, l, volatile=False):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(_l) for _l in l]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else: 
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

