import os

import torch
import torch.optim as optim
import numpy as np
import random 
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.io import loadmat
from model import DMNet
from train_model_DMLPA import train_model
from load_data import get_loader
from evaluate import fx_calc_map_label, fx_cal_acc_label
from config import args
from torch.backends import cudnn

######################################################################
# Start running

if __name__ == '__main__':
    # environmental setting: setting the following parameters based on your experimental environment.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    dataset = args.data_name  # 'MIRFLICKR25K' or 'NUS-WIDE-TC21' or 'MS-COCO' 'wiki' 'xmedianet' 'pascal'
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
        

    print('...Data loading is beginning...')

    data_loader, input_data_par = get_loader(dataset, args.batch_size)

    print('...Data loading is completed...')

    # model_ft = DMNet(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
    #                      num_classes=input_data_par['num_class'], t=t, k=k, inp=inp, GNN=gnn, n_layers=n_layers).cuda()
    model_ft = DMNet(in_dims=input_data_par['in_dims'], out_dim=args.out_dim, wv_matrix=input_data_par['wv_matrix'], num_fc_img=args.num_fc_img, num_fc_txt=args.num_fc_txt, bn=args.bn, dropout=args.dropout).cuda()
    print(model_ft)
    
    params_to_update = list(model_ft.parameters())

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(params_to_update, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epochs)
    if args.eval:
        model_ft.load_state_dict(torch.load('model/DALGNN_' + dataset + '.pth'))
    else:
        print('...Training is beginning...')
        # Train and evaluate                                          model, data_loaders, optimizer, args.view_name, num_epochs=500model, acc_history, epoch_loss_history
        model_ft, acc_hist, loss_hist = train_model(model_ft, data_loader, optimizer, lr_scheduler, args, wv_matrix=input_data_par['wv_matrix'])
        print('...Training is completed...')

        torch.save(model_ft.state_dict(), 'model/DALGNN_' + dataset + '.pth')

    print('...Evaluation on testing data...')
    model_ft.eval()
    with torch.no_grad():
        outputs = model_ft([torch.tensor(i).cuda() for i in input_data_par['data_test']])
    labels = input_data_par['label_test']
    # view1_feature, view2_feature =  view1_feature[0: 5000], view2_feature[0: 5000]
    # cos = view1_feature.mm(view2_feature.t()) / (view1_feature.norm(dim=-1).view([-1, 1]) * view2_feature.norm(dim=-1).view([1, -1]) + 1e-7)
    features = [f.detach().cpu().numpy() for f in outputs]
    
    results, str_out = [], '....Test:       '
    best_ret = 0
    for i in range(len(acc_hist)):
        if sum(acc_hist[i]) > best_ret:
            best_ret = sum(acc_hist[i])
            best_i = i
    
    best_val, val_str, idx = acc_hist[best_i], '....Validation: ', 0
    evaluate_method = fx_calc_map_label if args.eval_metric == 'map' else fx_cal_acc_label
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:
                results.append(fx_calc_map_label(features[i], features[j], labels[i], labels[j]))
                str_out = str_out + ('%s2%s: %.4f\t' % (args.view_name[i], args.view_name[j], results[-1]))
                val_str = val_str + ('%s2%s: %.4f\t' % (args.view_name[i], args.view_name[j], best_val[idx]))
                idx += 1
    
    val_str = '...{}, Average MAP = {}'.format(val_str, (sum(best_val) / len(best_val)))
    str_out = '...{}, Average MAP = {}'.format(str_out, (sum(results) / len(results)))
    print('Best epoch %d' % best_i)
    print(val_str)
    print(str_out)
    print(args)
    
    with open('results/%s.txt' % args.data_name, 'a+') as f:
        f.writelines([str(args), '\n', 'Best epoch %d' % best_i, '\n', val_str, '\n', str_out, '\n\n'])