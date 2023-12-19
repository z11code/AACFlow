import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils
import time
from termcolor import colored
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import pandas as pd
import torch.nn.functional as F
from attention_augmented_conv import AugmentedConv


def position_encoding(seqs):
    """
    Position encoding features introduced in "Attention is all your need",
    the b is changed to 1000 for the short length of sequence.
    """
    d = 128
    b = 1000
    res = []
    for seq in seqs:
        N = len(seq)
        value = []
        for pos in range(N):
            tmp = []
            for i in range(d // 2):
                tmp.append(pos / (b ** (2 * i / d)))
            value.append(tmp)
        value = np.array(value)
        pos_encoding = np.zeros((N, d))
        pos_encoding[:, 0::2] = np.sin(value[:, :])
        pos_encoding[:, 1::2] = np.cos(value[:, :])
        res.append(pos_encoding)
    return np.array(res)


def data_construct(seqs, labels, train):
    # Amino acid dictionary
    '''
    aa_dict = {'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5, 'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
               'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15, 'O': 16, 'S': 17, 'U': 18, 'T': 19,
               'W': 20, 'Y': 21, 'V': 22, 'X': 23}
    '''
    aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
             'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X':21}

    longest_num = len(max(seqs, key=len))
    sequences = [i.ljust(longest_num, 'X') for i in seqs]


    pep_codes = []
    for pep in seqs:
        current_pep = []
        for aa in pep:
            current_pep.append(aa_dict[aa])
        pep_codes.append(torch.tensor(current_pep))

    embed_data = rnn_utils.pad_sequence(pep_codes, batch_first=True)
    dataset = Data.TensorDataset(embed_data,
                                 torch.LongTensor(labels))
    batch_size = 15
    data_iter = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_iter



def load_bench_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=True)
    train_iter = [x for i, x in enumerate(data_iter) if i % 5 != 0]
    test_iter = [x for i, x in enumerate(data_iter) if i % 5 == 0]

    return train_iter, test_iter


def load_ind_data(file):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = tmp[0].values.tolist(), tmp[1].values.tolist()
    data_iter = data_construct(seqs, labels, train=False)
    return data_iter


class Flow_Attention(nn.Module):
    # flow attention in normal version
    def __init__(self, d_input, d_model, d_output, n_heads, drop_out=0.05, eps=5e-4):
        super(Flow_Attention, self).__init__()
        self.n_heads = n_heads
        self.query_projection = nn.Linear(d_input, d_model)
        self.key_projection = nn.Linear(d_input, d_model)
        self.value_projection = nn.Linear(d_input, d_model)
        self.out_projection = nn.Linear(d_model, d_output)
        self.dropout = nn.Dropout(drop_out)
        self.eps = eps

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)

        sink_incoming = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + self.eps, keys.sum(dim=2) + self.eps))
        source_outgoing = 1.0 / (torch.einsum("nhld,nhd->nhl", keys + self.eps, queries.sum(dim=2) + self.eps))

        conserved_sink = torch.einsum("nhld,nhd->nhl", queries + self.eps,
                                      (keys * source_outgoing[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.einsum("nhld,nhd->nhl", keys + self.eps,
                                        (queries * sink_incoming[:, :, :, None]).sum(dim=2) + self.eps)
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)  # for stability

        sink_allocation = torch.sigmoid(conserved_sink * (float(queries.shape[2]) / float(keys.shape[2])))
        source_competition = torch.softmax(conserved_source, dim=-1) * float(keys.shape[2])

        x = (self.dot_product(queries * sink_incoming[:, :, :, None],  # for value normalization
                              keys,
                              values * source_competition[:, :, :, None])  # competition
             * sink_allocation[:, :, :, None]).transpose(1, 2)  # allocation

        x = x.reshape(B, L, -1)
        x = self.out_projection(x)
        x = self.dropout(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        conv_kernel_size = 8
        pool_kernel_size = 4

        self.conv_net1 = nn.Sequential(
            nn.Conv1d(100, 32, kernel_size=conv_kernel_size, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=conv_kernel_size, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=2),
            nn.BatchNorm1d(32))
        self.conv_net2 = nn.Sequential(
            nn.Conv1d(32, 48, kernel_size=5, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(48, 48, kernel_size=5, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size, padding=2),
            nn.BatchNorm1d(48),
            nn.Dropout(p=0.2))
        self.conv_net3 = nn.Sequential(
            nn.Conv1d(48, 96, kernel_size=3, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=3, stride=1, padding=4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(96),
            nn.Dropout(p=0.2))

        self.attn_normal = Flow_Attention(76, 96, 44, 4)
        self.classifier1 = nn.Sequential(
            nn.Linear(4224, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.2),)

        self.classifier2 = nn.Sequential(
            nn.Linear(256, 48),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(48),
            nn.Dropout(p=0.2),)
        self.classifier3 = nn.Sequential(
            nn.Linear(48, 2),
            #nn.Sigmoid()
            nn.Softmax(dim=1))

        self.AAC = AugmentedConv(1, 100, 10, dk=40, dv=4, Nh=1, relative=False, stride=1)


    def forward(self, x):
        x = F.one_hot(x, num_classes=21)
        x = x.float()
        x = x.squeeze(1)
        out1 = x.view(-1, 50*21)
        x = x.reshape(x.size(0), 1, x.size(1), x.size(2))
        x = F.relu(self.AAC(x))
        x = x.view(-1, 100, 49*20)
        out2 = x.view(-1, 100*49*20)
        x = self.conv_net1(x)
        x = self.conv_net2(x)
        x = self.conv_net3(x)
        out3 = x.view(x.size(0), -1)
        out = self.attn_normal(x, x, x)
        out4 = out.view(out.size(0), 4224)
        out = self.classifier1(out4)
        out5 = self.classifier2(out)
        out = self.classifier3(out5)

        return out, out1, out2, out3, out4, out5


def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    rep_list = []
    for x, y in data_iter:
        outputs, out1, out2, out3, out4, out5 = net(x)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_pred = label_pred + outputs.argmax(dim=1).tolist()
        label_real = label_real + y.tolist()
    performance, Fpr, Tpr, precision, recall = caculate_metric(pred_prob, label_pred, label_real)
    return performance, Fpr, Tpr, rep_list, label_real, precision, recall

def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)


    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    performance = [ACC, Sensitivity, Specificity, AUC, MCC, precision, recall]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, FPR, TPR, precision, recall


def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lambda = 0.0
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss


def K_CV(file, k):
    tmp = pd.read_csv(file, header=None)
    seqs, labels = np.array(tmp[0].values.tolist()), np.array(tmp[1].values.tolist())
    data_iter = data_construct(seqs, labels, train=True)
    data_iter = list(data_iter)
    CV_perform = []
    for iter_k in range(k):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)
        train_iter = [x for i, x in enumerate(data_iter) if i % k != iter_k]
        test_iter = [x for i, x in enumerate(data_iter) if i % k == iter_k]
        #performance, _, ROC, PRC = train_test(train_iter, test_iter)
        performance, _ = train_test(train_iter, test_iter)
        print("k = " + str(iter_k + 1) +'reslut:',performance)
        CV_perform.append(performance)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ',
                                    'red') + '=' * 16 + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC,\tPre,\tRecall]\n')
    for out in np.array(CV_perform):
        print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(out[0], out[2], out[1], out[3], out[4]))
    mean_out = np.array(CV_perform).mean(axis=0)
    print('\n' + '=' * 16 + "Mean out" + '=' * 16)
    print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(mean_out[0], mean_out[2], mean_out[1], mean_out[3],
                                                              mean_out[4]))
    print('\n' + '=' * 60)

def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == '__main__':
    '''k-fold cross-validation'''
    K_CV("ACP_Train.csv", 5)
