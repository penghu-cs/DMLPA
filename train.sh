lr=1e-3
wd=0
max_epochs=200
interval=1
out_dim=1024
alpha1=0.5
alpha2=0.05
tau=0.5
metric='euclidean' # euclidean, cosine, mahalanobis, manhattan(cityblock), braycurtis, canberra, correlation, chebyshev
loss='mse' # mse mae kl kl_i kl_s bce
gpu=1
dropout=0.
bn=''

# nMSAD
# data_name='nMSAD'
# batch_size=100
# interval=1
# num_fc_img=0
# num_fc_txt=0
# max_epochs=200
# bn='--bn'

# pascal
data_name='pascal'
batch_size=100
interval=1
num_fc_img=2
num_fc_txt=1
max_epochs=100
bn='--bn'

# # MS-COCO
# data_name='MS-COCO'
# batch_size=1000
# num_fc_img=2
# num_fc_txt=1
# interval=1
# lr=5e-05
# bn='--bn'

# xmedianet
# data_name='xmedianet'
# batch_size=1000
# num_fc_img=3
# num_fc_txt=2
# bn='--bn'

python main.py --data_name $data_name --lr $lr --wd $wd --batch_size $batch_size --max_epochs $max_epochs --interval $interval --out_dim $out_dim --alpha1 $alpha1 --alpha2 $alpha2 --tau $tau --metric $metric --loss $loss --gpu $gpu --num_fc_img $num_fc_img --num_fc_txt $num_fc_txt --dropout $dropout $bn
