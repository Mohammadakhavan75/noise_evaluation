import torch
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from get_models import get_models
from datetime import datetime
import os
from models.wrn_ssnd import *
import dataset_loader
from sklearn.preprocessing import label_binarize


def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                         help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, default=0,
                         help='num_workers.')
    parser.add_argument('--start_epoch', type=int, default=0,
                         help='starting epoch from.')
    parser.add_argument('--save_path', type=str, default=None,
                         help='Path to save files.')
    parser.add_argument('--model_path', type=str, default=None,
                         help='Path to model to resume training.')
    parser.add_argument('--mode', type=str, default='classification',
                         help='must be mode of how model load.')
    # Optimizer Config
    parser.add_argument('--optimizer', type=str, default='sgd',
                         help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.001,
                         help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float, default=5,
                         help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=0.9,
                         help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--momentum', type=float, default=0.9,
                         help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005,
                         help='Weight decay (L2 penalty).')
    
    parser.add_argument('--run_index', default=0, type=int,
                         help='run index')
    # model config
    parser.add_argument('--model_name', default='resnet18', type=str,
                         help='give model name like resnet18|resnet34|vit')
    # parser.add_argument('--layers', default=[10], type=list, help='give model last layers as list like [512, 128, 10]')
    parser.add_argument('--pretrained', default=False, type=bool,
                         help='Use imagenet pretrained model')

    parser.add_argument('--layers', default=40, type=int,
                        help='total number of layers')
    parser.add_argument('--widen-factor', default=2, type=int,
                         help='widen factor')
    parser.add_argument('--droprate', default=0.3, type=float,
                        help='dropout probability')
    
    parser.add_argument('--device', default='cuda', type=str,
                         help='Use cpu or cuda')
    parser.add_argument('--T', default=1., type=float,
                         help='Tempreture of energy score')
    parser.add_argument('--shift', default='gaussian_noise', type=str,
                         help='Use cpu or cuda')
    args = parser.parse_args()

    return args


def tensor_to_np(x):
    return list(x.data.cpu().numpy())


def init_model(args):
    
    # get_model = get_models(args.model_name, args.layers)
    # model = get_model.get_model(args)
    if args.mode == 'binary':
        model = WideResNet(args.layers, 1, args.widen_factor, dropRate=args.droprate)
    elif args.mode == 'classification':
        model = WideResNet(args.layers, 10, args.widen_factor, dropRate=args.droprate)
    else:
        raise NotImplemented("mode must be classification or binary!!")

    if args.model_path is not None:
        print("Loading from pretrain!")
        model.load_state_dict(torch.load(args.model_path))

    if args.mode == 'binary':
        criterion = torch.nn.BCEWithLogitsLoss().to(args.device)
    elif args.mode == 'classification':
        criterion = torch.nn.CrossEntropyLoss().to(args.device)

    return model, criterion


def test(in_loader, out_loader, net, criterion, args):

    print('validation...')
    eval_loss = []
    eval_acc = []
    eval_auc = []
    net = net.to(args.device)
    net.eval()
    loader = zip(in_loader, out_loader)
    with torch.no_grad():
        for data_in, data_out in tqdm(loader):

            inputs_in, targets_in = data_in
            inputs_out, targets_out = data_out
            targets_in_b = np.asarray([0 for _ in range(len(inputs_in))])
            targets_out_b = np.asarray([1 for _ in range(len(inputs_out))])

            np_inputs = np.concatenate((inputs_in, inputs_out), axis=0)
            np_targets = np.concatenate((targets_in_b, targets_out_b), axis=0)

            inputs, targets = torch.tensor(np_inputs), torch.tensor(np.asarray(np_targets))
            inputs , targets = inputs.to(args.device) , targets.to(args.device)

            preds = net(inputs)
            
            # if args.mode == 'binary'
            # loss = criterion(preds, targets)

            targets = targets.unsqueeze(1).to(float)
            score_in = tensor_to_np(-(args.T * torch.logsumexp(preds[:len(inputs_in)] / args.T, dim=1)))
            score_out = tensor_to_np(-(args.T * torch.logsumexp(preds[len(inputs_in):] / args.T, dim=1)))
            auroc = compute_auroc(np.array(score_out), np.array(score_in))
            
            # acc = accuracy_score(tensor_to_np(torch.argmax(preds[:len(inputs_in)], axis=1)), tensor_to_np(targets_in))
            
            # eval_loss.append(loss.item())
            # eval_acc.append(acc)
            eval_auc.append(auroc)


    return eval_loss, eval_acc, eval_auc


def compute_auroc(out_scores, in_scores):
    in_labels = np.zeros(len(in_scores))
    out_labels = np.ones(len(out_scores))
    y_true = np.concatenate([in_labels, out_labels])
    y_score = np.concatenate([in_scores, out_scores])
    auroc = roc_auc_score(y_true=y_true, y_score=y_score)

    return auroc



if __name__ == '__main__':
    args = parsing()
    torch.manual_seed(args.seed)

    model, criterion = init_model(args)

    np_test_img_path = f'/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_test/{args.shift}.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_test/labels.npy'
    svhn_path = '/storage/users/makhavan/CSI/exp05/data/'


    # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    # std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform = transforms.ToTensor()
    
    in_test = dataset_loader.load_np_dataset(np_test_img_path, np_test_target_path, transform=transform)
    out_train, out_test = dataset_loader.load_svhn(svhn_path, transform=transform)

    in_loader = DataLoader(in_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    out_loader = DataLoader(out_test, batch_size=args.batch_size, num_workers=args.num_workers,  shuffle=False)

    
    train_global_iter = 0
    global_eval_iter = 0
    best_acc = 0.0

    eval_loss, eval_acc, eval_auc = test(in_loader, out_loader, model, criterion, args)

    print(f"Evaluation/avg_auc: {np.mean(eval_auc)}")

    # print(f"\nEvaluation/avg_loss: {np.mean(eval_loss)}",\
    #     f"Evaluation/avg_acc: {np.mean(eval_acc)}\n",\
    #     f"Evaluation/avg_auc: {np.mean(eval_auc)}")

