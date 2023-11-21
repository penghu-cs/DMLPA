import numpy as np
import scipy
import scipy.spatial
import torch

def fx_cal_acc_label(image, text, img_label, txt_label, k=0, dist_method='cosine'):
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    neigh.fit(text, txt_label)
    la = neigh.predict(image)
    return np.sum((la == img_label.reshape([-1])).astype(int)) / float(la.shape[0])

def fx_calc_map_label(image, text, img_label, txt_label, k=0, dist_method='cosine'):
    image, text, img_label, txt_label = torch.tensor(image).float().cuda(), torch.tensor(text).float().cuda(), torch.tensor(img_label).float().cuda(), torch.tensor(txt_label).float().cuda()
    if dist_method == 'euclidean':
        # dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
        dist_func = lambda i, t: ((i ** 2.).sum(-1).view([-1, 1]) + (t ** 2.).sum(-1).view([1, -1])) - 2. * i.mm(t.t())
    elif dist_method == 'cosine':
        # dist = scipy.spatial.distance.cdist(image, text, 'cosine')
        dist_func = lambda i, t: 1. - i.mm(t.t()) / (i.norm(dim=-1).view([-1, 1]) * t.norm(dim=-1).view([1, -1]))
    batch_size, ord, numcases = 5000, [], image.shape[0]
    for i in range(int(np.ceil(float(numcases) / batch_size))):
        dist = dist_func(image[i * batch_size: (i + 1) * batch_size], text)
        ord.append(dist.argsort().detach().cpu().numpy())
    del dist
    ord = np.concatenate(ord)
    sim = (img_label.mm(txt_label.t()) > 0).float().detach().cpu().numpy()
    tindex = np.arange(numcases, dtype=float) + 1
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        sim[i] = sim[i][order]
        num = max(sim[i].sum(), 1e-7)
        a = np.where(sim[i]==1)[0]
        sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
        res += [(sim[i] / tindex).sum() / num]

    return (sum(res) / len(res)).item()
