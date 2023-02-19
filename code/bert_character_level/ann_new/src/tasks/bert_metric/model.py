import random

import torch
import torch.nn as nn
import torch.nn.functional as nnf


class ContrastiveLoss(nn.Module):
    def __init__(self, margin, n_pos_pairs, n_neg_pairs):
        super().__init__()
        self.margin = margin
        self.n_pos_pairs = n_pos_pairs
        self.n_neg_pairs = n_neg_pairs

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        pd = nnf.pdist(embeddings)
        all_pairs = torch.combinations(labels)
        pair_labels = all_pairs[:, 0] == all_pairs[:, 1]

        pos_d, _ = torch.sort(pd[pair_labels == 1], descending=True)
        neg_d, _ = torch.sort(pd[pair_labels == 0], descending=False)
        pos_d = pos_d[:self.n_pos_pairs]
        neg_d = neg_d[:self.n_neg_pairs]

        loss = torch.cat((pos_d ** 2, nnf.relu(self.margin - neg_d) ** 2))
        return loss.mean()


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        pd = torch.cdist(embeddings, embeddings)
        size = labels.size(0)
        all_pairs = torch.combinations(torch.arange(size))
        pos_indices = labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]
        d_ap_s = []
        d_an_s = []
        for idx1, idx2 in all_pairs[pos_indices]:
            d_ap = pd[idx1, idx2]
            neg_mark = torch.ne(labels, labels[idx1])
            maxi = d_ap ** 2 + self.margin

            neg_indices = torch.where(torch.logical_and(
                neg_mark,
                torch.logical_and(torch.gt(pd[idx1], d_ap), torch.lt(pd[idx1] ** 2, maxi))
            ))[0]
            if neg_indices.size(0) > 0:
                d_ap_s.append(d_ap)
                d_an_s.append(pd[idx1, random.choice(neg_indices)])

            neg_indices = torch.where(torch.logical_and(
                neg_mark,
                torch.logical_and(torch.gt(pd[idx2], d_ap), torch.lt(pd[idx2] ** 2, maxi))
            ))[0]
            if neg_indices.size(0) > 0:
                d_ap_s.append(d_ap)
                d_an_s.append(pd[idx2, random.choice(neg_indices)])

        d_ap_s = torch.Tensor(d_ap_s)
        d_an_s = torch.Tensor(d_an_s)

        loss = torch.mean(d_ap_s ** 2 - d_an_s ** 2 + self.margin)
        return loss


class ClusterLoss(nn.Module):
    def __init__(self, alpha, mass_exp, q_exp):
        super().__init__()
        self.alpha = alpha
        self.mass_exp = mass_exp
        self.q_exp = q_exp

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, mass: torch.Tensor, size_map: dict):
        labels_to_idx = {}
        for idx, l in enumerate(labels):
            labels_to_idx.setdefault(l.item(), []).append(idx)

        centroids = []
        q = []
        loss_intra = 0
        mass = mass ** self.mass_exp
        for aff_id, idx_list in labels_to_idx.items():
            q.append(size_map[aff_id])
            cluster = embeddings[idx_list]
            cluster_mass = mass[idx_list]
            centroid = torch.sum(cluster * cluster_mass.unsqueeze(1), dim=0) / cluster_mass.sum()
            centroids.append(centroid)
            loss_intra += torch.sum((cluster - centroid) ** 2, dim=-1).mean()
        loss_intra /= len(labels_to_idx)

        pd = nnf.pdist(torch.vstack(centroids))
        q = torch.Tensor(q).to(embeddings.device) ** self.q_exp
        all_pairs = torch.combinations(torch.arange(len(centroids)))
        q1 = q[all_pairs[:, 0]]
        q2 = q[all_pairs[:, 1]]
        loss_inter = (q1 * q2 / pd).mean()

        loss = loss_intra + self.alpha * loss_inter
        return loss, loss_intra, self.alpha * loss_inter
