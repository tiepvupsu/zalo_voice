from __future__ import division
from split import my_split
# from sklearn.model_selection import train_test_split
# from myloss import SmoothLabel
from sklearn.metrics import accuracy_score
import config as cf
from nets import MyResNet
import utils
from data import build_dataloaders
import torch

import torch.nn as nn
import argparse
import copy
import random
from torchvision import transforms
# import time
import torch.backends.cudnn as cudnn
import os, sys
from time import time, strftime
import pdb
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
# from mytrain_test_split import mytrain_test_split_voice
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from predicts import singlemodel_class
# from nets import load_model, parallelize_model

parser = argparse.ArgumentParser(description='PyTorch Digital Mammography Training')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--net_type', default='resnet', type=str, help='model')
parser.add_argument('--depth', default=18, choices = [11, 16, 19, 18, 34, 50, 152, 161, 169, 121, 201], type=int, help='depth of model')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--finetune', '-f', action='store_true', help='Fine tune pretrained model')
parser.add_argument('--trainer', default='adam', type = str, help = 'optimizer')
parser.add_argument('--duration', default= 1.5, type = float, help='time duration for each file in second')
parser.add_argument('--n_tests', default=3, type = int, help='number of tests in valid set')
parser.add_argument('--gender', '-g', action='store_true', help='classify gender')
parser.add_argument('--accent', '-a', action='store_true', help='accent classifier')
parser.add_argument('--random_state', '-r', default = 2, type = int, help='random state in train_test_split')

parser.add_argument('--model_path', type=str, default = ' ')
parser.add_argument('--gamma', default = 0.5, type = float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--num_epochs', default=100, type=int,
                    help='Number of epochs in training')
parser.add_argument('--dropout_keep_prob', default=0.5, type=float)
parser.add_argument('--check_after', default=5,
                    type=int, help='check the network after check_after epoch')
parser.add_argument('--train_from', default=1,
                    choices=[0, 1],  # 0: from scratch, 1: from pretrained 1 (need model_path)
                    type=int,
                    help="training from beginning (1) or from the most recent ckpt (0)")

parser.add_argument('--frozen_until', '-fu', type=int, default = -1,
                    help="freeze until --frozen_util block")
parser.add_argument('--val_ratio', default=0.1, type=float, 
        help = "number of training samples per class")

########################################################################################33
if __name__ == '__main__':
    args = parser.parse_args()

    print('======================================================')
    print('Data preparation')
    dset_loaders, train_info = build_dataloaders(args)
    train_fns, semi_fns, val_fns, train_lbs, semi_lbs, val_lbs = train_info
    num_classes = len(set(train_lbs))

    def exp_lr_scheduler(args, optimizer, epoch):
        # after epoch 100, not more learning rate decay
        init_lr = args.lr
        lr_decay_epoch = 4 # decay lr after each 10 epoch
        weight_decay = args.weight_decay
        lr = init_lr * (0.6 ** (min(epoch, 200) // lr_decay_epoch)) 

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

        return optimizer, lr

    saved_models = './saved_model/'
    if not os.path.isdir(saved_models): os.mkdir(saved_models)
    saved_model_fn = saved_models + args.net_type + '_' +\
        str(args.depth) + '_' +  strftime('%m%d_%H%M') + '_r' + str(args.random_state) 
    print('model will be saved to {}'.format(saved_model_fn))
    print('********************************************************')
    old_model = './checkpoint/' + args.net_type + '_' + str(args.depth) + '_' +   args.model_path + '.t7'
    if args.train_from == 1 and os.path.isfile(old_model):
        print("| Load pretrained at  %s..." % old_model)
        checkpoint = torch.load(old_model, map_location=lambda storage, loc: storage)
        tmp = checkpoint['model']
        model = utils.unparallelize_model(tmp)
        try:
            top1acc = checkpoint['acc']
            print('previous acc\t%.4f'% top1acc)
        except KeyError:
            pass
        print('=============================================')
    else:
        model = MyResNet(args.depth, num_classes)

    model, optimizer = utils.net_frozen(args, model)
    model = utils.parallelize_model(model)
    criterion = nn.CrossEntropyLoss()
    ################################ 
    N_train = len(train_lbs)
    N_valid = len(val_lbs)
    best_acc = 0
    ########## Start training
    print('Start training ... ')
    t0 = time()
    for epoch in range(args.num_epochs):
        optimizer, lr = exp_lr_scheduler(args, optimizer, epoch) 
        print('#################################################################')
        print('=> Training Epoch #%d, LR=%.10f' % (epoch + 1, lr))
        running_loss, running_corrects, tot = 0.0, 0.0, 0.0
        running_loss_src, running_corrects_src, tot_src = 0.0, 0.0, 0.0
        ########################
        model.train()
        torch.set_grad_enabled(True)
        ## Training 
        for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['train']):
            optimizer.zero_grad()
            inputs = utils.cvt_to_gpu(inputs)
            labels = utils.cvt_to_gpu(labels)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            ############################################
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss.item()
            running_corrects += preds.eq(labels.data).cpu().sum()
            tot += labels.size(0)
            sys.stdout.write('\r')
            try:
                batch_loss = loss.item()
            except NameError:
                batch_loss = 0

            top1acc = float(running_corrects)/tot
            sys.stdout.write('| Epoch [%2d/%2d] Iter [%3d/%3d]\tBatch loss %.4f\tTop1acc %.4f'
                             % (epoch + 1, args.num_epochs, batch_idx + 1,
                                (len(train_fns) // args.batch_size), batch_loss/args.batch_size,
                                top1acc))
            sys.stdout.flush()
            sys.stdout.write('\r')

        top1acc =  float(running_corrects)/N_train
        epoch_loss = running_loss/N_train
        print('\n| Training loss %.8f\tTop1error %.4f'\
                % (epoch_loss, top1acc))

        utils.print_eta(t0, epoch, args.num_epochs)

        ###################################
        ## Validation
        if (epoch + 1) % args.check_after == 0:
            # Validation 
            ###################### 
            n_files = len(val_lbs)
            print('On test set')
            pred_output, pred_prob, _ = singlemodel_class(model, dset_loaders['test'], num_tests = 3)
            print(confusion_matrix(semi_lbs, pred_output))
            acc1 = accuracy_score(semi_lbs, pred_output)
            acc2 = accuracy_score(semi_lbs, pred_prob)
            print('acc_output: {}, acc_prob: {}'.format(acc1, acc2))
            print('On validation')
            pred_output, pred_prob, _ = singlemodel_class(model, dset_loaders['val'], num_tests =args.n_tests)
            print(confusion_matrix(val_lbs, pred_output))
            acc1 = accuracy_score(val_lbs, pred_output)
            acc2 = accuracy_score(val_lbs, pred_prob)
            print('acc_output: {}, acc_prob: {}'.format(acc1, acc2))
            ########### end test on multiple windows ##############3
            running_loss, running_corrects, tot = 0.0, 0.0, 0.0
            torch.set_grad_enabled(False)
            model.eval()
            for batch_idx, (inputs, labels, _) in enumerate(dset_loaders['val']):
                inputs = utils.cvt_to_gpu(inputs)
                labels = utils.cvt_to_gpu(labels)
                outputs = model(inputs)
                _, preds  = torch.max(outputs.data, 1)
                running_loss += loss.item()
                running_corrects += preds.eq(labels.data).cpu().sum()
                tot += labels.size(0)

            epoch_loss = running_loss / N_valid 
            top1acc= float(running_corrects)/N_valid
            # top3error = 1 - float(runnning_topk_corrects)/N_valid
            print('| Validation loss %.8f\tTop1acc %.4f'\
                    % (epoch_loss, top1acc))

            ################### save model based on best acc 
            if acc1 > best_acc:
                best_acc = acc1
                print('Saving model')
                best_model = copy.deepcopy(model)
                state = {
                    'model': best_model,
                    'acc' : acc1,
                    'clipped': args.duration,
                    'args': args
                }

                torch.save(state, saved_model_fn + '.t7')
                print('=======================================================================')
                print('model saved to %s' % (saved_model_fn + '.t7'))

