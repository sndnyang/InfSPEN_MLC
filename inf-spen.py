
# python 3.6
import os
import time
import pickle
import argparse

import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('data/bibtex/train.pickle', "rb") as f:
    temp = pickle.load(f)
    data_x = np.array([instance['feats'] for instance in temp])
    data_y = np.array([instance['types'] for instance in temp])

with open('data/bibtex/test.pickle', "rb") as f:
    temp = pickle.load(f)
    test_x = np.array([instance['feats'] for instance in temp])
    test_y = np.array([instance['types'] for instance in temp])

# the dataset is small, copy all to GPU
data_x = torch.FloatTensor(data_x).to(device)
data_y = torch.FloatTensor(data_y).to(device)
test_x = torch.FloatTensor(test_x).to(device)
test_y = torch.FloatTensor(test_y).to(device)


class MLP(nn.Module):
    # Base model/backbone, MLP
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(1836, 150)
        self.layer2 = nn.Linear(150, 150)
        self.out_l = nn.Linear(150, 159)

    def forward(self, x, only_feature_extraction=False):
        out = func.relu(self.layer1(x))
        out = func.relu(self.layer2(out))
        if not only_feature_extraction:
            out = self.out_l(out)
            out = torch.sigmoid(out)
        return out


def f1_map(y, pred):
    threshold = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
    best_f1 = 0
    for t in threshold:
        local_pred = pred > t
        local_f1 = f1_score(y.data.cpu().numpy(), local_pred.data.cpu().numpy(), average='samples')
        if local_f1 > best_f1:
            best_f1 = local_f1
    precision = np.mean(metrics.average_precision_score(
        y.data.cpu().numpy(), pred.data.cpu().numpy(), average=None
    ))

    return best_f1, precision


class EnergyNetwork(nn.Module):
    # Eq 1 and 2 in InfNet paper
    def __init__(self, weights_last_layer_mlp=150, label_dim=159, num_pairwise=16, non_linearity=nn.Softplus()):
        super(EnergyNetwork, self).__init__()

        self.non_linearity = non_linearity

        # equation 1 in inf spen paper
        self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        # in equation 2 of inf spen paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):

        # Local energy
        # equation 1 in inf spen paper
        e_local = torch.mm(x, self.B)
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        # e_local = e_local.view(e_local.size()[0], 1)

        # Label energy, equation 2
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label.view(-1), e_local)

        return e_global


class SPEN:
    def __init__(self, feature_net, energy_func, inference_net, params):
        self.feature_extractor = feature_net
        self.feature_extractor.eval()
        self.energy_func = energy_func
        self.inf_net = inference_net
        self.args = params

    def _compute_energy(self, inputs, targets):
        f_x = self.feature_extractor(inputs, only_feature_extraction=True)

        # Energy ground truth
        gt_energy = self.energy_func(f_x, targets)

        # Cost-augmented inference network
        pred_labels = self.inf_net(inputs)
        pred_energy = self.energy_func(f_x, pred_labels)
        return pred_labels, pred_energy, gt_energy

    def compute_loss(self, inputs, targets):

        pred_labels, pred_energy, gt_energy = self._compute_energy(inputs, targets)
        # Max-margin Loss
        pre_loss = torch.sum((pred_labels - targets)**2, dim=1) - pred_energy + gt_energy
        eneryg_loss = torch.max(pre_loss, torch.zeros_like(pre_loss))
        eneryg_loss = torch.mean(eneryg_loss)

        pred_y = pred_labels

        entropy_loss = - torch.mean(pred_y * torch.log(pred_y) + (1 - pred_y) * torch.log(1 - pred_y))

        inf_net_loss = -eneryg_loss + self.args.lamb_phi * sum(p.pow(2.0).sum() for p in self.inf_net.parameters()) \
            + self.args.lamb_pre_bias * sum((x - y).pow(2.0).sum() for x, y in zip(self.inf_net.state_dict().values(), self.feature_extractor.state_dict().values())) \
            + self.args.lamb_entropy * entropy_loss
        e_net_loss = eneryg_loss + self.args.lamb_theta * sum(p.pow(2.0).sum() for p in self.energy_func.parameters())
        return pred_labels, e_net_loss, inf_net_loss

    def pred(self, x):
        with torch.no_grad():
            y_pred = self.inf_net(x)
        return y_pred

    def inference(self, x, n_steps=1):

        sd = self.inf_net.state_dict()
        inf_net2 = MLP()
        inf_net2.load_state_dict(sd)
        self.inf_net.eval()
        optimizer = optim.Adam(self.inf_net.parameters(), lr=args.lr_psi, weight_decay=0)
        with torch.no_grad():
            y_pred = self.inf_net(x)

        self.inf_net.train()

        return y_pred


if __name__ == '__main__':
    args = argparse.Namespace()
    args.lamb_phi = 0.001
    args.lamb_entropy = 1
    args.lamb_pre_bias = 1
    args.lamb_theta = 0.01
    args.lr_theta = 0.001
    args.lr_phi = 0.001

    # Stage 1
    # pre-training feature net
    dataset = 'bibtex'
    base_model = MLP().to(device)
    if not os.path.isfile('../%s_base.pt' % dataset):
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3, weight_decay=0)

        for epoch in range(20):
            inds = np.random.permutation(list(range(len(data_x))))
            start_time = time.time()
            for i in range(0, 4880, 80):
                idx = inds[i:i + 80]
                data = data_x[idx]
                label = data_y[idx]
                logits = base_model(data)

                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            end_time = time.time()
            print('Epoch', epoch)
            print('Training Speed: %d examples/sec' % int(4880 / (end_time - start_time)))
            with torch.no_grad():
                logits = base_model(test_x)
                pred_test = torch.sigmoid(logits)

            f1, mAP = f1_map(test_y, pred_test)
            print(f1, mAP)
            print()
        torch.save(base_model.state_dict(), '../%s_base.pt' % dataset)
    else:
        base_model.load_state_dict(torch.load('../%s_base.pt' % dataset))

    with torch.no_grad():
        logits = base_model(test_x)
        pred_test = torch.sigmoid(logits)

    f1, mAP = f1_map(test_y, pred_test)
    print(f1, mAP)

    # Stage 2, train the inference net
    inf_net = MLP().to(device)
    inf_net.load_state_dict(base_model.state_dict())
    energy_net = EnergyNetwork(base_model.out_l.weight).to(device)

    optim_inf = torch.optim.Adam(inf_net.parameters(), lr=args.lr_phi, weight_decay=0)
    optim_energy = torch.optim.Adam(energy_net.parameters(), lr=args.lr_theta, weight_decay=0)

    spen = SPEN(base_model, energy_net, inf_net, args)

    for epoch in range(100):
        inds = np.random.permutation(list(range(len(data_x))))
        for i in range(0, 4880, 80):
            idx = inds[i:i + 80]
            data = data_x[idx]
            label = data_y[idx]

            optim_inf.zero_grad()
            _, _, inf_loss = spen.compute_loss(data, label)

            inf_loss.backward()
            optim_inf.step()

            optim_energy.zero_grad()
            _, e_loss, _ = spen.compute_loss(data, label)

            e_loss.backward()
            optim_energy.step()
        print(epoch)

        pred_test = spen.pred(test_x)
        f1, mAP = f1_map(test_y, pred_test)
        print(f1, mAP)

        # pred_test = spen.inference(test_x)
        # best_f1, mAP = f1_map(test_y, pred_test)
        # print(best_f1, mAP)
        print()
