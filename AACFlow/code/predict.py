from main import load_ind_data, Net, load_model, evaluate
import torch
from termcolor import colored
import pandas as pd


def predict(file):
    data_iter = load_ind_data(file)
    model = Net()
    path_pretrain_model = "Model.pt"
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        ind_performance, ind_roc_data, ind_prc_data, _, _, precision, recall = evaluate(data_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                   + '\n[ACC,\tSP,\t\tSE,\t\tAUC,\tMCC,\tPre,\tRecall]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            ind_performance[0], ind_performance[2], ind_performance[1], ind_performance[3],
            ind_performance[4]) + '\n' + '=' * 60

    return ind_results


ind_result = predict('ACP_Test.csv')
print(ind_result)