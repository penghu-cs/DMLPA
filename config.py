import argparse
# Training settings
parser = argparse.ArgumentParser(description='UCCH implementation')

#########################
#### data parameters ####
#########################
parser.add_argument("--data_name", type=str, default="nMSAD", help="data name")  # wiki, pascal, nus-wide, xmedianet, xmedianet_full, half_mnist, reuters, nus_wide_10_ml, CMPlaces, voc_data_ml, CMPlaces_raw, MNIST_SAD, noisyMNIST, MSCOCO, MSCOCO_doc2vec
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--interval', type=int, default=20)
parser.add_argument('--out_dim', type=int, default=1024, help='output shape')
parser.add_argument('--alpha1', type=float, default=.5)
parser.add_argument('--alpha2', type=float, default=.05)
parser.add_argument('--tau', type=float, default=.2)
parser.add_argument('--metric', type=str, default='euclidean') # euclidean, cosine, mahalanobis, manhattan(cityblock), braycurtis, canberra, correlation, chebyshev
parser.add_argument('--loss', type=str, default='mse') # mse mae kl kl_i kl_s bce 
parser.add_argument('--eval', action='store_true')
parser.add_argument('--gpu', type=str, default='0')


parser.add_argument('--num_fc_img', type=int, default=2)
parser.add_argument('--num_fc_txt', type=int, default=1)
parser.add_argument('--bn', action='store_true')
parser.add_argument('--dropout', type=float, default=0.)

args = parser.parse_args()
print(args)

args.eval_metric = 'acc'
if args.data_name in ['MS-COCO', 'wiki', 'xmedianet', 'pascal']:
    view_name = ['Img', 'Txt']
    args.eval_metric = 'map'
elif args.data_name in ['MNIST']:
    view_name = ['PV', 'NV']
elif args.data_name == 'nMSAD':
    view_name = ['SAD', 'V1', 'V2']
args.view_name = view_name
# python UCCH.py --data_name iapr_fea --bit 128 --alpha 0.4 --num_hiden_layers 3 3 --margin 0.2 --max_epochs 20 --train_batch_size 256 --shift 0.5 --lr 0.0001 --optimizer Adam --warmup_epoch 5