import os
import pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as func

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'


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


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1836, 150)
        self.layer2 = nn.Linear(150, 150)

    def forward(self, x):
        out = func.relu(self.layer1(x))
        out = func.relu(self.layer2(out))
        return out


class EnergyNet(nn.Module):
    def __init__(self, weights_last_layer_mlp=150, feature_dim=150, label_dim=159,
                 num_pairwise=16, non_linearity=nn.Softplus()):
        super().__init__()

        self.non_linearity = non_linearity

        self.linear_wt = nn.Linear(150, label_dim, bias=False)

        # Label energy terms, C1/c2  in equation 5 of SPEN paper
        self.C1 = nn.Linear(label_dim, num_pairwise)

        self.c2 = nn.Linear(num_pairwise, 1, bias=False)

    def forward(self, x, y):
        # Local energy
        negative_logits = self.linear_wt(x)
        feat_probs = torch.sigmoid(-1 * negative_logits)

        # element-wise product
        e_local = torch.mul(negative_logits, y)
        e_local = torch.sum(e_local, dim=1)

        # Label energy
        e_label = self.non_linearity(self.C1(y))
        e_label = self.c2(e_label)
        assert e_label.view(-1).shape[0] == e_label.shape[0]
        assert e_label.view(-1).shape[0] == e_local.shape[0]
        e_global = torch.add(e_label.view(-1), e_local.view(-1))

        return e_global, feat_probs


class InfNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1836, 150)
        self.layer2 = nn.Linear(150, 150)
        self.layer3 = nn.Linear(150, 159, bias=False)

    def forward(self, x):
        out = func.relu(self.layer1(x))
        out = func.relu(self.layer2(out))
        out = self.layer3(out)
        return torch.sigmoid(out), out


class SPEN():
    def __init__(self, feature_net, energy_net, inf_net, n_steps_inf=1, input_dim=1836, label_dim=159):
        self.feature_extractor = feature_net
        self.feature_extractor.eval()
        self.energy_net = energy_net
        self.inf_net = inf_net

        self.phi0 = InfNet().to(device)
        self.phi0.load_state_dict(inf_net.state_dict())

    def compute_loss(self, inputs, targets):
        f_x = self.feature_extractor(inputs).detach()

        # Energy ground truth
        gt_energy, _ = self.energy_net(f_x, targets)

        # Cost-augmented inference network
        pred_probs, logits = self.inf_net(inputs)

        pred_energy, _ = self.energy_net(f_x, pred_probs)

        # Max-margin Loss
        diff = torch.sum((pred_probs - targets)**2, dim=1)
        gt_en = gt_energy
        inf_en = pred_energy
        pre_loss_real = diff - inf_en + gt_en

        energy_loss = torch.relu(pre_loss_real)
        pre_loss_real = torch.mean(pre_loss_real)
        energy_loss = torch.mean(energy_loss)

        # entropy_loss = nn.BCELoss()(pred_probs, pred_probs.detach())
        entropy_loss = func.binary_cross_entropy_with_logits(logits, pred_probs.detach())

        reg_losses_phi = 0.5 * sum(p.pow(2.0).sum() for p in self.inf_net.parameters())

        pretrain_bias = sum((x - y).pow(2.0).sum() for x, y in zip(list(self.inf_net.parameters()), self.phi0.state_dict().values()))

        reg_losses_theta = 0.5 * sum(p.pow(2.0).sum() for p in self.energy_net.parameters())

        inf_net_loss = energy_loss - 0.001 * reg_losses_phi - 1 * pretrain_bias - 1 * entropy_loss

        inf_net_loss = -inf_net_loss

        e_net_loss = energy_loss + 0.001 * reg_losses_theta

        summaries = {
            'infer cost': inf_net_loss.item(),
            'energy cost': e_net_loss.item(),
            'base_objective': energy_loss.item(),
            'base_obj_real': pre_loss_real.item(),
            'energy_inf_net': pred_energy.mean().item(),
            'energy_ground_truth': gt_energy.mean().item(),
            'reg_losses_theta': reg_losses_theta.item(),
            'reg_losses_phi': reg_losses_phi.item(),
            'reg_losses_entropy': entropy_loss.item(),
            'pretrain_bias': pretrain_bias.item()
        }

        return pred_probs, e_net_loss, inf_net_loss, summaries

    def pred(self, x):
        with torch.no_grad():
            y_pred, _ = self.inf_net(x)
        return y_pred

    def inference_loss(self, x):

        f_x = self.feature_extractor(x)
        # inference network
        pred_probs, logits = self.inf_net(x)
        pred_energy, _ = self.energy_net(f_x, pred_probs)

        entropy_loss = func.binary_cross_entropy_with_logits(logits, pred_probs.detach())
        reg_losses_phi = 0.5 * sum(p.pow(2.0).sum() for p in self.inf_net.parameters())

        inf_net_loss = torch.mean(pred_energy) + 0.001 * reg_losses_phi + 1 * entropy_loss

        return inf_net_loss


def pretrain(feat_net, energy_net):

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(list(feat_net.parameters()) + list(energy_net.parameters()), lr=1e-3, weight_decay=0)

    for epoch in range(10):
        inds = np.random.permutation(list(range(len(data_x))))
        for i in range(0, 4880, 32):
            l = inds[i:i + 32]
            data = data_x[l]
            label = data_y[l]
            feat = feat_net(data)
            _, logits = energy_net(feat, label)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', epoch)
        with torch.no_grad():
            feat = feat_net(test_x)
            _, pred_test = energy_net(feat, test_y)

        best_f1, mAP = f1_map(test_y, pred_test, 0.5)
        print(best_f1, mAP)
        print()

    with torch.no_grad():
        feat = feat_net(test_x)
        _, pred_test = energy_net(feat, test_y)

    best_f1, mAP = f1_map(test_y, pred_test)
    print('feature net F1', best_f1, mAP)


def copy_net(feat_net, energy_net, inf_net):
    inf_net.layer1.weight.data = feat_net.layer1.weight.data.clone()
    inf_net.layer1.bias.data = feat_net.layer1.bias.data.clone()
    inf_net.layer2.weight.data = feat_net.layer2.weight.data.clone()
    inf_net.layer2.bias.data = feat_net.layer2.bias.data.clone()
    inf_net.layer3.weight.data = - energy_net.linear_wt.weight.data.clone()


def phrase_2(spen, feat_net, energy_net, inf_net):
    optim_inf = torch.optim.Adam(inf_net.parameters(), lr=1e-3, weight_decay=0)
    optim_energy = torch.optim.Adam(list(energy_net.C1.parameters()) + list(energy_net.c2.parameters()), lr=1e-3, weight_decay=0)
    optim_e = torch.optim.Adam(list(inf_net.parameters()) + list(energy_net.parameters()), lr=1e-3, weight_decay=0)

    pred_test = spen.pred(test_x)
    best_f1, mAP = f1_map(test_y, pred_test)
    print('inf net start', best_f1, mAP)

    for epoch in range(50):

        for j, i in enumerate(range(0, 4880, 32)):
            if i + 32 > 4880:
                i = 4880 - 32
            l = np.arange(i, i + 32)
            data = data_x[l]
            # print(data.sum())
            label = data_y[l]

            optim_e.zero_grad()
            optim_inf.zero_grad()
            preds, e_loss, inf_loss, summary = spen.compute_loss(data, label)

            inf_loss.backward()

            optim_inf.step()

            optim_e.zero_grad()
            optim_inf.zero_grad()
            preds, e_loss, inf_loss, summary = spen.compute_loss(data, label)

            e_loss.backward()

            optim_energy.step()

        print(epoch)

        pred_test = spen.pred(test_x)
        best_f1, mAP = f1_map(test_y, pred_test)
        print('current inf net', best_f1, mAP)

        with torch.no_grad():
            feat = feat_net(test_x)
            _, pred_test = energy_net(feat, test_y)

        f1, mAP = f1_map(test_y, pred_test, 0.5)
        print('feature net', f1, mAP)
        print()


def inference(spen):
    optimizer = torch.optim.Adam(spen.inf_net.parameters(), lr=0.00001, weight_decay=0)
    pred_test = spen.pred(test_x)
    best_f1, mAP = f1_map(test_y, pred_test)
    print('inf net start', best_f1, mAP)

    for epoch in range(10):

        for j, i in enumerate(range(0, 4880, 32)):
            if i + 32 > 4880:
                i = 0
            l = np.arange(i, i + 32)
            data = data_x[l]
            optimizer.zero_grad()
            inf_loss = spen.inference_loss(data)
            inf_loss.backward()
            optimizer.step()

        print(epoch)
        pred_test = spen.pred(test_x)
        best_f1, mAP = f1_map(test_y, pred_test)
        print('current inf net', best_f1, mAP)
        print()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

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

    feat_net = MLP().to(device)
    energy_net = EnergyNet().to(device)

    print("Phrase 1: start pretraining the feature net")
    pretrain(feat_net, energy_net)

    inf_net = InfNet().to(device)
    print("copy to inference net")
    copy_net(feat_net, energy_net, inf_net)

    spen = SPEN(feat_net, energy_net, inf_net)

    print("Phrase 2: optimize the inference net")
    phrase_2(spen, feat_net, energy_net, inf_net)

    print("Phrase 3: test time optimization")
    inference(spen)
