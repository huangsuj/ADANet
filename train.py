import os
import warnings
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from args import parameter_parser
from utils import *
from Dataloader import load_data
from model import Our
import copy
from torch.autograd import Variable
from Discriminator import Discriminator
import configparser

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()

    Graph_dataset = ['NGs']
    for index, item in enumerate(Graph_dataset):
        args.dataset = item
        print('--------------Multi-view Datasets: {}--------------------'.format(args.dataset))

        conf = configparser.ConfigParser()
        config_path = './config_demo' + '.ini'
        conf.read(config_path, encoding='utf-8')
        args.num_epoch = int(conf.getfloat(args.dataset, 'epoch'))
        args.lr = conf.getfloat(args.dataset, 'lr')
        args.weight_decay = conf.getfloat(args.dataset, 'weight_decay')
        args.dropout = conf.getfloat(args.dataset, 'dropout')
        args.alpha = conf.getfloat(args.dataset, 'alpha')
        if args.fix_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
        tab_printer(args)

        ### Load Data
        adj_list, fea_list, labels, train_mask, valid_mask, test_mask = load_data(args)
        n = len(labels)
        num_classes = len(np.unique(labels))
        num_view = len(adj_list)
        labels = labels.to(args.device)

        all_ACC = []
        all_F1 = []
        input_dims = []
        for i in range(num_view):
            input_dims.append(fea_list[i].shape[1])
        for n_num in range(args.n_repeated):

            ## Model initalize
            model = Our(n, input_dims, num_classes, args.dropout, args.hdim, args.device).to(args.device) ## Generator
            loss_function1 = torch.nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            D = Discriminator(num_classes).to(args.device)
            loss_fn = torch.nn.BCELoss()
            optimizer_D = torch.optim.Adam(D.parameters(), lr=0.001)

            best_valid_acc = 0
            Tanh = nn.Tanh()
            real_label = Variable(torch.ones(n, 1)).to(args.device)
            fake_label = Variable(torch.zeros(n, 1)).to(args.device)
            _, _, output_view_invariant_list, _ = model(fea_list, adj_list)  # Initialization
            with tqdm(total=args.num_epoch, desc="Training") as pbar:
                for epoch in range(args.num_epoch):
                    model.train()

                    for view in range(num_view):
                        loss_fake = 0
                        loss_G = 0

                        ### Discriminator
                        real_input = output_view_invariant_list[view].clone().detach()
                        real_out = D(real_input)
                        loss_real = loss_fn(real_out, real_label)
                        for j in range(num_view):
                            if j != view:
                                fake_out = output_view_invariant_list[j].clone().detach()
                                fake_out = D(fake_out)
                                loss_fake += loss_fn(fake_out, fake_label) / num_view
                        loss_D = loss_real + loss_fake
                        optimizer_D.zero_grad()
                        loss_D.backward()
                        optimizer_D.step()

                        ### Generator
                        MI_loss_list, rec_loss_list, output_view_invariant_list, output_view_specific_list = model(fea_list, adj_list)
                        invariant_output_softmax = F.log_softmax(output_view_invariant_list[view], dim=1)
                        loss_pos_GCN = loss_function1(invariant_output_softmax[train_mask], labels[train_mask])
                        ### Negative entropy loss
                        output = F.sigmoid(output_view_specific_list[view])
                        log_output = torch.log(output)
                        loss_neg_GCN = torch.sum(torch.sum(log_output[train_mask], dim=1), dim=0)
                        loss_neg_GCN = - loss_neg_GCN / log_output[train_mask].shape[0]
                        ### Generator loss
                        for g in range(num_view):
                            G_output_ = Tanh(output_view_invariant_list[g])
                            G_output_ = D(G_output_)
                            loss_G += loss_fn(G_output_, real_label) / num_view
                        ### Mutual Information
                        loss_mi = MI_loss_list[view]
                        ### Reconstructor loss
                        loss_rec = rec_loss_list[view]

                        loss_ = loss_pos_GCN + args.alpha * (loss_neg_GCN + loss_mi + loss_G + loss_rec)
                        optimizer.zero_grad()
                        loss_.backward()
                        optimizer.step()

                    pred_labels = torch.argmax(sum(output_view_invariant_list), 1).data.cpu().numpy()
                    Train_ACC, Train_F1, _, _ = get_evaluation_results(labels.cpu().detach().numpy()[train_mask],
                                                                pred_labels[train_mask])

                    with torch.no_grad():
                        model.eval()
                        MI_loss_list, rec_loss_list, output_view_invariant_list, output_view_specific_listt = model(fea_list, adj_list)
                        pred_labels = torch.argmax(sum(output_view_invariant_list), 1).cpu().detach().numpy()
                        Valid_ACC, F1, _, _ = get_evaluation_results(labels.cpu().detach().numpy()[valid_mask],
                                                               pred_labels[valid_mask])

                    pbar.set_postfix({
                        'Total_loss': '{:.4f}'.format(loss_.item()),
                        'Train_ACC': '{:.2f}'.format(Train_ACC * 100),
                        'Valid_ACC': '{:.2f}'.format(Valid_ACC * 100),
                    })
                    pbar.update(1)

                    if (Valid_ACC >= best_valid_acc):
                        best_valid_acc = Valid_ACC
                        best_model = copy.deepcopy(model)
                        best_epoch = epoch

                    if args.early_stop:
                        if (Valid_ACC >= best_valid_acc):
                            patience = args.patience
                        else:
                            patience -= 1
                            if (patience < 0):
                                print("Early Stopped!")
                                break

                test_model = best_model
                with torch.no_grad():
                    test_model.eval()
                    _, _, output_view_invariant_list, _ = test_model(fea_list, adj_list)
                    print("Evaluating the model")
                    pred_labels = torch.argmax(sum(output_view_invariant_list), 1).cpu().detach().numpy()
                    ACC, F1, _, _ = get_evaluation_results(labels.cpu().detach().numpy()[test_mask],
                                                           pred_labels[test_mask])
                    print("ACC: {:.2f}, F1: {:.2f}".format(ACC * 100, F1 * 100))

                all_ACC.append(ACC)
                all_F1.append(F1)
        print("-----------------------")
        print("ACC: {:.2f} ({:.2f})".format(np.mean(all_ACC) * 100, np.std(all_ACC) * 100))
        print("F1 : {:.2f} ({:.2f})".format(np.mean(all_F1) * 100, np.std(all_F1) * 100))
        print("-----------------------")