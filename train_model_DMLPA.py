from __future__ import print_function
from __future__ import division
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torchvision

from evaluate import fx_calc_map_label
from loss import DMLPA_loss


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def train_model(model, data_loaders, optimizer, lr_scheduler, args, wv_matrix=None):
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_history = []
    epoch_loss_history = []
    

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(args.max_epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.max_epochs))
        print('-' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Set model to training mode
                model.train()
                running_loss = 0.0
                # Iterate over data.
                for inputs, labels in data_loaders[phase]:
                    if sum([torch.sum(i != i) for i in inputs]) > 1:
                        print("Data contains Nan.")

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        # Get model outputs and calculate loss
                        # Special case for inception because in training it has an auxiliary output. In train
                        #   mode we calculate the loss by summing the final output and the auxiliary output
                        #   but in testing we only consider the final output.
                        if torch.cuda.is_available():
                            inputs = [d.cuda() for d in inputs]
                            labels = [l.float().cuda() for l in labels]

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward
                        outs = model(inputs)

                        dmlpa_loss = DMLPA_loss(outs, inputs, labels, metric=args.metric, tau=args.tau, alpha1=args.alpha1, alpha2=args.alpha2, wv_matrix=wv_matrix, loss=args.loss)
                        loss = dmlpa_loss
                        
                        import pdb
                        # pdb.set_trace()

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item()

                lr_scheduler.step()
                epoch_loss = running_loss / len(data_loaders[phase])
                print('Train Loss: {:.7f}'.format(epoch_loss))
            elif phase == 'valid' and (epoch + 1) % args.interval == 0:
                # Set model to evaluate mode
                model.eval()
                t_data, t_labels = [], []
                with torch.no_grad():
                    for inputs, labels in data_loaders['valid']:
                        if torch.cuda.is_available():
                            inputs = [d.cuda() for d in inputs]
                            labels = [l.cuda() for l in labels]
                        outs = model(inputs)
                        for i in range(len(inputs)):
                            if len(t_data) == 0:
                                t_data = [[] for _ in range(len(inputs))]
                                t_labels = [[] for _ in range(len(inputs))]
                            t_data[i].append(outs[i].cpu().numpy())
                            t_labels[i].append(labels[i].cpu().numpy())

                t_data = [np.concatenate(t) for t in t_data]
                t_labels = [np.concatenate(t) for t in t_labels]

                results, str_out = [], '{} Loss: {:.7f}\t'.format(phase, epoch_loss)
                for i in range(len(t_data)):
                    for j in range(len(t_data)):
                        if i != j:
                            results.append(fx_calc_map_label(t_data[i], t_data[j], t_labels[i], t_labels[j]))
                            # import pdb
                            # pdb.set_trace()
                            str_out = str_out + ('%s2%s: %.4f\t' % (args.view_name[i], args.view_name[j], results[-1]))

                ds_len = float(len(data_loaders[phase].dataset))

                print(str_out)
                acc_history.append(results)
                epoch_loss_history.append(epoch_loss)
                
                # deep copy the model
                if sum(results) / len(results) > best_acc:
                    best_acc = sum(results) / len(results)
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best average ACC: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, epoch_loss_history
