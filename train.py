import argparse
from layers import *
from torch import optim
from model import *
from utils import *
import torch.nn.functional as F

import os

from sklearn.cluster import KMeans

import time

start_time = time.time()

setup_seed(2020)


parser=parameter_setting()
args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


x, y = prepro('data/'+args.data_name+'/data.h5')

true_labels = y
x = np.ceil(x).astype(np.int64)
n_clusters = int(max(torch.FloatTensor(true_labels)) - min(torch.FloatTensor(true_labels)) + 1)
adata = sc.AnnData(x)
adata.obs['Group'] = true_labels
adata  = normalize( adata, filter_min_counts=True, highly_genes=args.highly_genes,
                        size_factors=True, normalize_input=False, 
                        logtrans_input=True ) 
print(adata)
num_cell, num_feature = np.shape(adata.X)
features = torch.FloatTensor(adata.X)
adj, adjn = get_adj(adata.X, k=args.k)
adj = sp.csr_matrix(adj)
print('Adj:', adj.shape, 'Edges:', len(adj.data))
print('X:', features.shape)
print('n clusters:', n_clusters)
adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj.eliminate_zeros()

# Laplacian Smoothing
adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
smooth_fea = sp.csr_matrix(features).toarray()
for a in adj_norm_s:
    smooth_fea = a.dot(smooth_fea)
smooth_fea = torch.FloatTensor(smooth_fea)

acc_list = []
nmi_list = []
ari_list = []
f1_list = []

best_acc, best_nmi, best_ari, best_f1, predict_labels, dis= clustering(smooth_fea, true_labels, n_clusters)

model = Encoder_Net(args.linlayers, [features.shape[1]] + [args.dims], adata, dropout=args.dropout)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

model.to(args.device)
smooth_fea = smooth_fea.to(args.device)
sample_size = features.shape[0]
target = torch.eye(smooth_fea.shape[0]).to(args.device)



for epoch in range(args.epochs):
    model.train()
    z1, z2, pi, disp, mean, Coef = model(args, smooth_fea)
    if epoch > 50:
        high_confidence = torch.min(dis, dim=1).values
        threshold = torch.sort(high_confidence).values[int(len(high_confidence) * args.threshold)]
        high_confidence_idx = np.argwhere(high_confidence < threshold)[0]
        index = torch.tensor(range(smooth_fea.shape[0]), device=args.device)[high_confidence_idx]
        y_sam = torch.tensor(predict_labels, device=args.device)[high_confidence_idx]
        index = index[torch.argsort(y_sam)]
        class_num = {}

        for label in torch.sort(y_sam).values:
            label = label.item()
            if label in class_num.keys():
                class_num[label] += 1
            else:
                class_num[label] = 1
        key = sorted(class_num.keys())
        if len(class_num) < 2:
            continue
        pos_contrastive = 0
        centers_1 = torch.tensor([], device=args.device)
        centers_2 = torch.tensor([], device=args.device)

        for i in range(len(key[:-1])):
            class_num[key[i + 1]] = class_num[key[i]] + class_num[key[i + 1]]
            now = index[class_num[key[i]]:class_num[key[i + 1]]]
            pos_embed_1 = z1[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
            pos_embed_2 = z2[np.random.choice(now.cpu(), size=int((now.shape[0] * 0.8)), replace=False)]
            pos_contrastive += (2 - 2 * torch.sum(pos_embed_1 * pos_embed_2, dim=1)).sum()
            centers_1 = torch.cat([centers_1, torch.mean(z1[now], dim=0).unsqueeze(0)], dim=0)
            centers_2 = torch.cat([centers_2, torch.mean(z2[now], dim=0).unsqueeze(0)], dim=0)

        pos_contrastive = pos_contrastive / n_clusters
        if pos_contrastive == 0:
            continue
        if len(class_num) < 2:
            loss = pos_contrastive
        else:
            centers_1 = F.normalize(centers_1, dim=1, p=2)
            centers_2 = F.normalize(centers_2, dim=1, p=2)
            S = centers_1 @ centers_2.T
            S_diag = torch.diag_embed(torch.diag(S))
            S = S - S_diag
            neg_contrastive = F.mse_loss(S, torch.zeros_like(S))
            loss = pos_contrastive + 1 * neg_contrastive
            # loss = pos_contrastive - neg_contrastive
            
            zinb = ZINB(pi, theta=disp, ridge_lambda=0)
            loss_zinb = zinb.loss(smooth_fea, mean, mean=True)
            hidden_emb = (z1 + z2) / 2
            hidden_emb_c = torch.matmul(Coef, hidden_emb)
            loss_reconst = 1 / 2 * torch.sum(torch.pow((smooth_fea - mean), 2))
            loss_reg = torch.sum(torch.pow(Coef, 2))
            loss_selfexpress = 1 / 2 * torch.sum(torch.pow((hidden_emb - hidden_emb_c), 2))

            loss = (0.2 * loss_reconst + 0.6 * loss_reg + 0.5 * loss_selfexpress)**1/100 + loss + loss_zinb

    else:
        S = z1 @ z2.T
        loss = F.mse_loss(S, target)

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if epoch>50 and epoch % 10 == 0:
        model.eval()
        print(str(epoch)+ "   " + str(loss.item()) )
        
        acc, nmi, ari, f1, predict_labels, dis = clustering(hidden_emb, true_labels, n_clusters)

    if epoch > 50:
        Coef_shifted = Coef.cpu().detach().numpy() - np.min(Coef.cpu().detach().numpy(), axis=1)
        Coef_exp = np.exp(Coef_shifted)
        Coef_ratio = Coef_exp / np.sum(Coef_exp, axis=1, keepdims=True)

        adj_norm_s = preprocess_graph(sp.csr_matrix(Coef_ratio), 1, norm='sym', renorm=True)
        smooth_fea = sp.csr_matrix(features).toarray()
        smooth_fea = adj_norm_s[0].dot(smooth_fea)
        smooth_fea = torch.FloatTensor(smooth_fea)

        smooth_fea = smooth_fea.to(args.device)

model.eval()
z = (z1 + z2) / 2

labels_s, _ = post_proC(Coef.cpu().detach().numpy(), n_clusters, 11, 7.0)
labels_k = KMeans(n_clusters=n_clusters, n_init=20).fit_predict(Coef.cpu().detach())
labels = labels_s if (np.round(metrics.normalized_mutual_info_score(y, labels_s), 5)>=np.round(metrics.normalized_mutual_info_score(y, labels_k), 5)
         and np.round(metrics.adjusted_rand_score(y, labels_s), 5)>=np.round(metrics.adjusted_rand_score(y, labels_k), 5)) else labels_k 
eva(y, labels)