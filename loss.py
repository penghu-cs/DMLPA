import torch
import torch.nn as nn
import torch.nn.functional as F

criterion_md = nn.CrossEntropyLoss()


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def cla_loss(view1_predict, view2_predict, labels_1, labels_2):
    cla_loss1 = ((view1_predict - labels_1.float()) ** 2).sum(1).sqrt().mean()
    cla_loss2 = ((view2_predict - labels_2.float()) ** 2).sum(1).sqrt().mean()

    return cla_loss1 + cla_loss2


def mdl_loss(view1_feature, view2_feature, labels_1, labels_2):
    cos = lambda x, y: x.mm(y.t()) / (
        (x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1, labels_1).float()
    Sim12 = calc_label_sim(labels_1, labels_2).float()
    Sim22 = calc_label_sim(labels_2, labels_2).float()
    term11 = ((1 + torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term12 = ((1 + torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term22 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    mdl_loss = term11 + term12 + term22

    return mdl_loss


def gan_loss(view1_modal_view1, view2_modal_view1, view1_modal_view2, view2_modal_view2):
    bs = view1_modal_view1.size()[0]
    img_md = torch.ones(bs, dtype=torch.long).cuda()
    txt_md = torch.zeros(bs, dtype=torch.long).cuda()
    return criterion_md(view1_modal_view1, img_md) + criterion_md(view2_modal_view1, txt_md) + \
           criterion_md(view1_modal_view2, img_md) + criterion_md(view2_modal_view2, txt_md)


def soft_con_loss(view1_feature, view2_feature, labels, t=0.21, gamma=0.13):
    view1_feature = F.normalize(view1_feature, dim=1)
    view2_feature = F.normalize(view2_feature, dim=1)
    # cosine similarity: NxN
    sim_view12 = torch.matmul(view1_feature, view2_feature.T) / t
    sim_view11 = torch.matmul(view1_feature, view1_feature.T) / t
    sim_view22 = torch.matmul(view2_feature, view2_feature.T) / t
    #label_L1 = labels.sum(1)
    #label_sim = torch.matmul(labels, labels.T) / (label_L1[None, :] + label_L1[:, None] - torch.matmul(labels, labels.T))
    label_sim = torch.matmul(labels, labels.T).clamp(max=1.0)
    #label_sim = label_sim ** 0.5
    pro_inter = label_sim / label_sim.sum(1, keepdim=True).clamp(min=1e-6)
    label_sim_intra = (label_sim - torch.eye(label_sim.shape[0]).cuda()).clamp(min=0)
    pro_intra = label_sim_intra / label_sim_intra.sum(1, keepdim=True).clamp(min=1e-6)

    # logits: NxN
    logits_view12 = sim_view12 - torch.log(torch.exp(1.06 * sim_view12).sum(1, keepdim=True))
    logits_view21 = sim_view12.T - torch.log(torch.exp(1.06 * sim_view12.T).sum(1, keepdim=True))
    logits_view11 = sim_view11 - torch.log(torch.exp(1.06 * sim_view11).sum(1, keepdim=True))
    logits_view22 = sim_view22 - torch.log(torch.exp(1.06 * sim_view22).sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos_view12 = (pro_inter * logits_view12).sum(1)
    mean_log_prob_pos_view21 = (pro_inter * logits_view21).sum(1)
    mean_log_prob_pos_view11 = (pro_intra * logits_view11).sum(1)
    mean_log_prob_pos_view22 = (pro_intra * logits_view22).sum(1)

    # supervised cross-modal contrastive loss
    loss = - mean_log_prob_pos_view12.mean() - mean_log_prob_pos_view21.mean() \
           - gamma * (mean_log_prob_pos_view11.mean() + mean_log_prob_pos_view22.mean())

    return loss



def my_dist(data, metric="euclidean"):
    if metric == "euclidean":
        if data.shape[0] > 200:
            norm = (data ** 2.).sum(-1, keepdim=True)
            dist = ((norm + norm.t()) - 2. * data.mm(data.t())).abs()
        else:
            dist = data.unsqueeze(1) - data.unsqueeze(0)
            dist = (dist ** 2).sum(-1)
    elif metric == "cosine":
        dist = 1. - (data.mm(data.t()) / ((data ** 2).sum(1).sqrt().view([-1, 1]).mm((data ** 2).sum(1).sqrt().view([1, -1]))).clamp(min=1e-6))
        # dist = dist ** 2
    elif metric == 'mahalanobis':
        m = data.mean(0, keepdim=True)
        S = data.t().mm(m).inverse()
        dist = ((data - m).mm(S).mm(data - m))
        dist = dist ** 2
    elif metric == 'cityblock':
        dist = data.unsqueeze(1) - data.unsqueeze(0)
        dist = dist.abs().sum(-1)
        dist = dist ** 2
    elif metric == 'manhattan':
        dist = data.unsqueeze(1) - data.unsqueeze(0)
        dist = dist.abs().sum(-1)
        dist = dist ** 2
    elif metric == 'chebyshev':
        dist = data.unsqueeze(1) - data.unsqueeze(0)
        dist = dist.abs().max(-1)[0]
        dist = dist ** 2
    elif metric == 'braycurtis':
        dist1 = (data.unsqueeze(1) - data.unsqueeze(0)).abs().sum(-1)
        dist2 = (data.unsqueeze(1) + data.unsqueeze(0)).abs().sum(-1)
        dist = dist1 / dist2
        dist = dist ** 2
    elif metric == 'canberra':
        dist1 = (data.unsqueeze(1) - data.unsqueeze(0)).abs()
        dist2 = (data.unsqueeze(1).abs() + data.unsqueeze(0).abs())
        dist = (dist1 / dist2).sum(-1)
        dist = dist ** 2
    elif metric == 'correlation':
        data = data - data.mean(0, keepdim=True)
        dist = 1. - (data.mm(data.t()) / ((data ** 2).sum(1).sqrt().view([-1, 1]).mm((data ** 2).sum(1).sqrt().view([1, -1]))).clamp_min(0.6))
    return dist

def DMLPA_loss(features, input_data, train_y, metric="euclidean", tau=1., alpha1=0.5, alpha2=0.05, wv_matrix=None, loss='mse'):
    # metric: euclidean, cosine, mahalanobis
    features = [F.normalize(feat, dim=1) for feat in features]
    # input_data = [F.normalize(torch.tensor(wv_matrix[d.cpu()].sum(1, keepdims=True)).cuda().reshape([d.shape[0], -1]) if d.dtype is torch.int64 else d, dim=1) for d in input_data]
    input_data = [F.normalize((torch.tensor(wv_matrix[d.cpu()].sum(1, keepdims=True)).cuda() / (d != d.max()).sum(1, keepdim=True).clamp_min(1e-7)).reshape([d.shape[0], -1]) if d.dtype is torch.int64 else d.reshape([d.shape[0], -1]), dim=1) for d in input_data]
    # input_data = [(torch.tensor(wv_matrix[d.cpu()].sum(1, keepdims=True)).cuda() / (d != d.max()).sum(1, keepdim=True).clamp_min(1e-7)).reshape([d.shape[0], -1]) if d.dtype is torch.int64 else d.reshape([d.shape[0], -1]) for d in input_data]
 
    n_view, map_list = len(features), []
    for v in range(n_view):
        feat = input_data[v]
        # dist_tmp = my_dist(feat, metric)
        dist_tmp = my_dist(feat, metric)
        bool_inx = (train_y[v].mm(train_y[v].t()) > 0).float()

        same_dist = dist_tmp * bool_inx
        delta_a = same_dist.max() * alpha1
        delta_a = 1. if delta_a < 1. else delta_a
        same_dist /= delta_a

        between_inx = 1. - bool_inx
        between_dist = dist_tmp * between_inx.float()
        delta_b = dist_tmp[between_inx > 0].min() * alpha2
        delta_b = 1e-4 if delta_b < 1e-4 else delta_b
        between_dist /= delta_b

        map_list.append((same_dist + between_dist).detach())

    # loss = utils.DMLPA_loss(gc_list, train_y, train_dist_map, n_view, metric=metric, tau=tau)

    dest = []
    for vi in range(n_view):
        tmp = []
        for vj in range(n_view):
            bool_inx = train_y[vi].mm(train_y[vj].t()).float()
            bool_inx = (bool_inx > 0.5).float()
            if vi == vj:
                tmp.append((-map_list[vi]).exp())
            else:
                bool_inx1 = 1. - bool_inx
                bool_inx2 = 1. - bool_inx
            
                dist_min_i = ((map_list[vi] + bool_inx1 * map_list[vi].max()).topk(2, largest=False)[0][:, 1]).reshape([-1, 1])
                dist_min_j = ((map_list[vj] + bool_inx2 * map_list[vj].max()).topk(2, largest=False)[0][:, 1]).reshape([1, -1])
                
                dist_min_ij = (-((dist_min_i > dist_min_j).float() * dist_min_j + (dist_min_i <= dist_min_j).float() * dist_min_i)).exp()
                dist_min_ij.fill_diagonal_(1.)
                same_dist = bool_inx * dist_min_ij

                diff_inx1 = 1. - bool_inx
                diff_inx2 = 1. - bool_inx

                diff_inx = 1. - bool_inx
                dist_max_i = (map_list[vi] * diff_inx1).max(1)[0].reshape([-1, 1])
                dist_max_j = (map_list[vj] * diff_inx2).max(1)[0].reshape([1, -1])
                dist_max_ij = (-((dist_max_i > dist_max_j).float() * dist_max_i + (dist_max_i <= dist_max_j).float() * dist_max_j)).exp()
                diff_dist = diff_inx * dist_max_ij

                dist = same_dist + diff_dist
                tmp.append(dist)
                # tmp.append(bool_inx)
        dest.append(torch.cat(tmp, dim=1))
    dest = torch.cat(dest, dim=0)
    
    D = dest.sum(1)
    D = (1. / D.clamp_min(1e-10)).diag()
    dest = D.mm(dest)
    
    dist = my_dist(torch.cat(features), metric)
    P = (-dist / tau).exp()
    D = P.sum(1)
    D = (1. / D.clamp_min(1e-10)).diag()
    P = D.mm(P)
    
    if loss == 'mse':
        loss_func = lambda x, y: (x - y).pow(2.).sum(1).mean()
    elif loss == 'mae':
        loss_func = lambda x, y: (x - y).abs().sum(1).mean()
    elif loss == 'kl':
        loss_func = lambda x, y: (x * (x / y.clamp_min(1e-7)).clamp_min(1e-7)).sum(1).mean()
    elif loss == 'kl_i':
        loss_func = lambda x, y: (y * (y / x.clamp_min(1e-7)).clamp_min(1e-7)).sum(1).mean()
    elif loss == 'kl_s':
        loss_func = lambda x, y: (y * (y / x.clamp_min(1e-7)).clamp_min(1e-7) + x * (x / y.clamp_min(1e-7)).clamp_min(1e-7)).sum(1).mean()
    elif loss == 'bce':
        loss_func = lambda x, y: -(y * x.clamp_min(1e-7).log() + (1 - y) * (1 - x).clamp_min(1e-7).log()).sum(1).mean()
    else:
        raise ValueError('Unknown loss!')
    return loss_func(P, dest)
    
