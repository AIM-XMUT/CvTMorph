import os, glob, torch, numpy as np
from torchvision import transforms
from torch import optim
import torch.nn as nn
import models, losses
from generators import load_volfile, scan_to_scan, get_volfile_list, patient_in_scan_to_scan
import time
import nibabel as nib
import utils


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    steps_per_epoch = 100
    save_model_dir = "./save_model_pt/"

    # subject to subject
    train_dir = "/data2/guoyifan/myData/TrainSet/vols/"
    train_list = glob.glob(train_dir + "*.nii.gz")
    train_set_gen = scan_to_scan(train_list)

    """
    train_dir = "/data2/guoyifan/myData/4D-Lung/Patients/"
    # our experiment train way
    train_list = []
    for num in range(0, 15):
        train_list.append(train_dir + "Patient_" + str(num) + "/")
    train_set_gen = patient_in_scan_to_scan(train_list)
    """

    lr = 0.0001
    epoch_start = 0
    max_epoch = 100
    reg_model = models.register_model((160, 160, 160), 'nearest')
    reg_model.cuda()

    model = models.HyBirdCvTUNet(img_size=(160, 160, 160))

    cont_training = False
    if cont_training:
        epoch_start = 200
        load_model_pt = "./save_model_pt/"
        lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        model = torch.load(load_model_pt)
    else:
        pass

    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    criterions = [losses.NCC(), losses.Grad3d(penalty='l2')]
    weights = [1.0, 1.0]
    lossall = np.zeros((1, max_epoch))

    print("Training Starts:")
    for epoch in range(epoch_start, max_epoch):

        loss_all = AverageMeter()
        start_time = time.time()
        for idx in range(steps_per_epoch):
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = next(train_set_gen)
            data = [torch.from_numpy(t).cuda() for t in data]
            x = data[0]
            y = data[1]
            x = x[np.newaxis, np.newaxis, ...]  # [1, 1, 160, 160, 160]
            y = y[np.newaxis, np.newaxis, ...]
            x = x.float()
            y = y.float()

            model_in = torch.cat((x, y), dim=1)
            model_out = model(model_in)

            loss = 0
            loss_vals = []
            curr_loss = criterions[0](model_out[0], y) * weights[0]
            loss_vals.append(curr_loss)
            loss += curr_loss
            curr_loss = criterions[1](model_out[1], y) * weights[1]
            loss_vals.append(curr_loss)
            loss += curr_loss

            loss_all.update(loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            del model_in
            del model_out

            loss = 0
            model_in = torch.cat((y, x), dim=1)
            model_out = model(model_in)

            curr_loss = criterions[0](model_out[0], x) * weights[0]
            loss_vals[0] += curr_loss
            loss += curr_loss
            curr_loss = criterions[1](model_out[1], x) * weights[1]
            loss_vals[1] += curr_loss
            loss += curr_loss

            loss_all.update(loss.item(), y.numel())
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            """
            if idx % 10 == 0:
                print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}'.format(idx, steps_per_epoch,
                                                                                       loss.item(),
                                                                                       loss_vals[0].item() / 2,
                                                                                       loss_vals[1].item() / 2,
                                                                                       ))
            """
        lossall[:, epoch] = np.array([loss_all.avg])
        if epoch % 50 == 0:
            torch.save(model, os.path.join(save_model_dir, '%04d.pt' % epoch))
        end_time = time.time()
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg), "cost time:", end_time - start_time)
    torch.save(model, os.path.join(save_model_dir, '%04d.pt' % max_epoch))
    print("Training end!")
    np.save(save_model_dir + 'loss.npy', lossall)


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


if __name__ == "__main__":
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
