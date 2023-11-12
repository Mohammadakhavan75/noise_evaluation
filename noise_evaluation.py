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

def parsing():
    parser = argparse.ArgumentParser(description='Tunes a CIFAR Classifier with OE',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=64, help='Batch size.')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for np(tinyimages80M sampling); 1|2|8|100|107')
    parser.add_argument('--num_workers', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--start_epoch', type=int, 
                        default=0, help='starting epoch from.')
    parser.add_argument('--save_path', type=str, 
                        default=None, help='Path to save files.')
    parser.add_argument('--model_path', type=str, 
                        default=None, help='Path to model to resume training.')
    # Optimizer Config
    parser.add_argument('--optimizer', type=str,
                        default='sgd', help='The initial learning rate.')
    parser.add_argument('--learning_rate', '-lr', type=float,
                        default=0.001, help='The initial learning rate.')
    parser.add_argument('--lr_update_rate', type=float,
                        default=5, help='The update rate for learning rate.')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.9, help='The gamma param for updating learning rate.')
                        
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float,
                        default=0.0005, help='Weight decay (L2 penalty).')

    parser.add_argument('--run_index', default=0, type=int, help='run index')
    # model config
    parser.add_argument('--model_name', default='resnet18', type=str, help='give model name like resnet18|resnet34|vit')
    parser.add_argument('--layers', default=[10], type=list, help='give model last layers as list like [512, 128, 10]')
    args = parser.parse_args()

    return args


def tensor_to_np(x):
    return list(x.data.cpu().numpy())


def train(train_loader, net, train_global_iter, criterion, optimizer, device):

    print("traning...")
    net = net.to(device)
    net.train()  # enter train mode

    # track train classification accuracy
    epoch_accuracies = []
    epoch_loss = []
    for data_in in tqdm(train_loader):
        
        inputs, targets = data_in
        inputs , targets = inputs.to(device) , targets.to(device)

        optimizer.zero_grad()
        preds = net(inputs)

        loss = criterion(preds, targets)
        
        acc = accuracy_score(list(tensor_to_np(torch.argmax(preds, axis=1))), list(tensor_to_np(targets)))
        epoch_accuracies.append(acc)
        epoch_loss.append(loss.item())

        train_global_iter += 1
        writer.add_scalar("Train/loss", loss.item(), train_global_iter)
        writer.add_scalar("Train/acc", acc, train_global_iter)

        loss.backward()
        optimizer.step()

    return train_global_iter, epoch_loss, epoch_accuracies 


def test(val_loader, net, global_eval_iter, criterion, device):
    print('validation...')
    eval_loss = []
    eval_acc = []
    eval_auc = []
    net = net.to(device)
    net.eval()
    with torch.no_grad():
        for data_in in tqdm(val_loader):

            inputs, targets = data_in
            inputs , targets = inputs.to(device) , targets.to(device)

            preds = net(inputs)

            loss = criterion(preds, targets)
            auc = roc_auc_score(tensor_to_np(targets), tensor_to_np(preds))
            acc = accuracy_score(tensor_to_np(torch.argmax(preds, axis=1)), tensor_to_np(targets))
            
            eval_loss.append(loss.item())
            eval_acc.append(acc)
            eval_auc.append(auc)

            writer.add_scalar("Evaluation/loss", loss.item(), global_eval_iter)
            writer.add_scalar("Evaluation/acc", acc, global_eval_iter)
            writer.add_scalar("Evaluation/auc", auc, global_eval_iter)

            global_eval_iter += 1

    return global_eval_iter, eval_loss, eval_acc, eval_auc


def init_model(args):
    
    get_model = get_models(args.model_name, args.layers)
    model = get_model.get_model()

    if args.model_path is not None:
        print("Loading from pretrain")
        model.load_state_dict(torch.load(args.model_path))

    model = model.to(args.device)



    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, 
                                    momentum=args.momentum,weight_decay=args.decay, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.decay)
    else:
        raise NotImplemented("Not implemented optimizer!")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_update_rate, gamma=args.lr_gamma)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)


    return model, criterion, optimizer, scheduler



if __name__ == '__main__':
    args = parsing()
    torch.manual_seed(args.seed)

    model, criterion, optimizer, scheduler = init_model(args)


    cifar10_path = '../data/'
    svhn_path = '../data/'
    np_train_img_path = '/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_train/gaussian_noise.npy'
    np_train_target_path = '/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_train/labels.npy'
    np_test_img_path = '/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_train/gaussian_noise.npy'
    np_test_target_path = '/storage/users/makhavan/CSI/exp04/data_aug/CorCIFAR10_train/labels.npy'

    # np_train = dataset_loader.load_np_dataset(np_train_img_path, np_train_target_path)
    # np_test = dataset_loader.load_np_dataset(np_test_img_path, np_test_target_path)

    # cifar_train, cifar_test = dataset_loader.load_cifar10(cifar10_path)

    # train_dataset = dataset_loader.combiner([np_train, cifar_train])
    # test_dataset = dataset_loader.combiner([np_test, cifar_test])

    # train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    # val_loader = DataLoader(test_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # out_train, out_test = dataset_loader.load_svhn(svhn_path)
    # out_train_loader = DataLoader(out_train, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    # out_val_loader = DataLoader(out_test, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # Download and load CIFAR-10 dataset
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)




    if args.model_path is not None:
        save_path = args.save_path
        model_save_path = save_path + 'models/'
    else:
        addr = datetime.today().strftime('%Y-%m-%d-%H-%M-%S-%f')
        save_path = './run/exp-' + addr + f"_({args.run_index})_" + f'_lr_{args.learning_rate}' + f'_lrur_{args.lr_update_rate}' + f'_lrg_{args.lr_gamma}' + f'_{args.optimizer}' + '/'
        model_save_path = save_path + 'models/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path, exist_ok=True)

    writer = SummaryWriter(save_path)

    train_global_iter = 0
    global_eval_iter = 0
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        print('epoch', epoch + 1, '/', args.epochs)
        train_global_iter, epoch_loss, epoch_accuracies = train(train_loader, model, train_global_iter, criterion, optimizer, args.device)
        global_eval_iter, eval_loss, eval_acc, eval_auc = test(test_loader, model, global_eval_iter, criterion, args.device)


        writer.add_scalar("Train/avg_loss", np.mean(epoch_loss), epoch)
        writer.add_scalar("Train/avg_acc", np.mean(epoch_accuracies), epoch)
        writer.add_scalar("Evaluation/avg_loss", np.mean(eval_loss), epoch)
        writer.add_scalar("Evaluation/avg_acc", np.mean(eval_acc), epoch)
        # writer.add_scalar("Evaluation/avg_auc", np.mean(eval_auc), epoch)

        print(f"\n\nTrain/avg_loss: {np.mean(epoch_loss)} Train/avg_acc: {np.mean(epoch_accuracies)} \
            Evaluation/avg_loss: {np.mean(eval_loss)} Evaluation/avg_acc: {np.mean(eval_acc)}\n\n")
        
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path,f'model_params_epoch_{epoch}.pt'))

        if np.mean(eval_acc) > best_acc:
            best_acc = np.mean(eval_acc)
            torch.save(model.state_dict(), os.path.join(save_path,'best_params.pt'))
        
        if np.mean(eval_acc) < best_acc:
            scheduler.step()

