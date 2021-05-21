import os

import pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as func
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
with open('data/bibtex/train.pickle', "rb") as f:
    temp = pickle.load(f)
    data_x = np.array([instance['feats'] for instance in temp])
    data_y = np.array([instance['types'] for instance in temp])

with open('data/bibtex/test.pickle', "rb") as f:
    temp = pickle.load(f)
    test_x = np.array([instance['feats'] for instance in temp])
    test_y = np.array([instance['types'] for instance in temp])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_x = torch.FloatTensor(data_x).to(device)
data_y = torch.FloatTensor(data_y).to(device)
test_x = torch.FloatTensor(test_x).to(device)
test_y = torch.FloatTensor(test_y).to(device)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1836, 150)
        self.layer2 = nn.Linear(150, 150)
        self.out_l = nn.Linear(150, 159, bias=False)

    def forward(self, x, only_feature_extraction=False):
        out = func.relu(self.layer1(x))
        out = func.relu(self.layer2(out))
        if not only_feature_extraction:
            out = self.out_l(out)
            out = torch.sigmoid(out)
        return out


def f1_map(y, pred, threshold=None):
    if threshold is None:
        threshold = [0.05, 0.10, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]
    else:
        threshold = [0.5]
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
    def __init__(self, weights_last_layer_mlp=150, feature_dim=150, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
        super().__init__()

        self.non_linearity = non_linearity

        self.B = torch.nn.Parameter(torch.transpose(-weights_last_layer_mlp, 0, 1))

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = torch.nn.Parameter(torch.empty(label_dim, num_pairwise))
        torch.nn.init.normal_(self.C1, mean=0, std=np.sqrt(2.0 / label_dim))

        self.c2 = torch.nn.Parameter(torch.empty(num_pairwise, 1))
        torch.nn.init.normal_(self.c2, mean=0, std=np.sqrt(2.0 / num_pairwise))

    def forward(self, x, y):
        # Local energy
        e_local = torch.mm(x, self.B)
        # element-wise product
        e_local = torch.mul(y, e_local)
        e_local = torch.sum(e_local, dim=1)
        # e_local = e_local.view(e_local.size()[0], 1)

        # Label energy
        e_label = self.non_linearity(torch.mm(y, self.C1))
        e_label = torch.mm(e_label, self.c2)
        e_global = torch.add(e_label.view(-1), e_local)

        return e_global


class SPEN:
    def __init__(self, feature_net, energy_net, inf_net, label_dim=159, args=None):
        super().__init__()
        self.feature_extractor = feature_net
        self.feature_extractor.eval()
        self.energy_net = energy_net
        self.inf_net = inf_net
        self.hidden_size = 150

        self.cost_infnet = nn.Sequential(nn.Linear(2 * label_dim, label_dim), nn.Softmax(dim=1)).to(device)
        # self.local2label = nn.Linear(self.hidden_size, self.num_tags)
        # self.inf2label = nn.Linear(self.hidden_size, self.num_tags)

    def _compute_energy(self, inputs, targets):
        f_x = self.feature_extractor(inputs, only_feature_extraction=True)

        # Energy ground truth
        gt_energy = self.energy_net(f_x, targets)

        # inference network
        inf_labels = self.inf_net(inputs)
        inf_energy = self.energy_net(f_x, inf_labels)

        temp = torch.cat((inf_labels, targets), dim=1)
        cost_inf_y = self.cost_infnet(temp)
        cost_inf_energy = self.energy_net(f_x, cost_inf_y)

        return inf_labels, inf_energy, cost_inf_y, cost_inf_energy, gt_energy

    def forward(self, inputs, targets):
        inf_labels, inf_energy, cost_inf_y, cost_inf_energy, gt_energy = self._compute_energy(inputs, targets)

        delta = torch.sum( (cost_inf_y - targets)**2, dim=1)

        local_ce = nn.BCELoss()(inf_labels, targets)

        # Max-margin Loss for theta
        hinge_loss_1 = delta - cost_inf_energy + gt_energy
        cost_loss_1 = torch.mean(torch.relu(hinge_loss_1))

        hinge_loss_2 = - inf_energy + gt_energy
        cost_loss_2 = torch.mean(torch.relu(hinge_loss_2))

        energy_loss = cost_loss_1 + cost_loss_2

        # loss for inf and cost inf
        inf_loss = torch.mean(-delta + inf_energy + cost_inf_energy)  # + 0.1 * local_ce
        return inf_labels, cost_inf_y, energy_loss, inf_loss

    def pred(self, x):
        with torch.no_grad():
            y_pred = self.inf_net(x)
        return y_pred


import time
# Stage 1
# pre-training feature net or load pretrained
dataset = 'bibtex'
base_model = MLP().to(device)
if not os.path.isfile('../%s_basej.pt' % dataset):
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
            pred_test = base_model(test_x)

        f1, mAP = f1_map(test_y, pred_test)
        print(f1, mAP)
        print()
    torch.save(base_model.state_dict(), '../%s_basej.pt' % dataset)
else:
    base_model.load_state_dict(torch.load('../%s_basej.pt' % dataset))

with torch.no_grad():
    pred_test = base_model(test_x)

f1, mAP = f1_map(test_y, pred_test)
print("current base model", f1, mAP)

# Stage 2, train the inference net
inf_net = MLP().to(device)
inf_net.load_state_dict(base_model.state_dict())

with torch.no_grad():
    pred_test = inf_net(test_x)

f1, mAP = f1_map(test_y, pred_test)

f1s = [f1]
maps = [mAP]
print("init inference model", f1, mAP)

energy_net = EnergyNetwork(base_model.out_l.weight).to(device)

spen = SPEN(base_model, energy_net, inf_net,)

optim_inf = torch.optim.Adam(list(inf_net.parameters()) + list(spen.cost_infnet.parameters()), lr=1e-3, weight_decay=0)
optim_energy = torch.optim.Adam(energy_net.parameters(), lr=1e-3, weight_decay=0)

K = 5
print("K", K)
epochs = 100
for epoch in range(epochs):
    inds = np.random.permutation(list(range(len(data_x))))
    for i in range(0, 4880, 32):
        l = inds[i:i+32]
        data = data_x[l]
        label = data_y[l]

        for i in range(K):
            optim_inf.zero_grad()
            preds, _, e_loss, inf_loss = spen.forward(data, label)
            inf_loss.backward()
            optim_inf.step()

        optim_energy.zero_grad()
        preds, _, e_loss, inf_loss = spen.forward(data, label)

        e_loss.backward()
        optim_energy.step()
    print(epoch)

    pred_test = spen.pred(test_x)
    best_f1, mAP = f1_map(test_y, pred_test)
    print(best_f1, mAP)
    f1s.append(best_f1)
    maps.append(mAP)


plt.plot(np.arange(epochs + 1), f1s)
plt.xlabel('Epoch')
plt.ylabel('F1')
plt.title('F1 joint K=%d' % K)
plt.show()
plt.close()
