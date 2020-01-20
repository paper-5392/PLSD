import numpy as np
import pandas as pd
import torch
import time
import os

import utils
import config
from plsd import PLSD


def main(file_path, ratio, n_run, n_selected_inl, batch_size=64, n_epoch=50, lr=0.1, seed=-1, test_percentage=0.4):
    data_name = file_path.split("/")[-1].split(".")[0]
    print("-----------------------%s, %d*%d, %.4f-----------------------" %
          (data_name, batch_size, n_epoch, lr))

    r_auc_roc, r_auc_pr, r_time = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
    for i in range(n_run):
        print(">>>>>>%s, Round %d START" % (data_name, i+1))
        df = pd.read_csv(file_path)
        x = df.values[:, :-1]
        labels = np.array(df.values[:, -1], dtype=int)
        dim = x.shape[1]
        if dim > 500:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"

        x_train, y_train, x_test, y_test = utils.split_train_test(x, labels, test_size=test_percentage, seed=seed)
        semi_y_train = utils.semi_setting(x_train, y_train, ratio, seed)

        print("x_train #obj:", x_train.shape[0], ", x_train #dimension:", x_train.shape[1])
        print("#True Outliers:", np.sum(y_train))
        print("#known Outliers:", np.sum(semi_y_train))

        s_time = time.time()

        plsd = PLSD(device=device, name=data_name, seed=seed)
        plsd.fit(x_train, semi_y_train, batch_size=batch_size, n_epochs=n_epoch, lr=lr)
        y_score = plsd.predict(x_test, y_test, n_selected_inl)

        r_time[i] = time.time()-s_time
        r_auc_roc[i], r_auc_pr[i] = utils.get_performance(y_score, y_test)
        print(">>>>>>%s, Round %d END: AUC-ROC: %.4f, AUC-PR: %.4f, %.4fs" %
              (data_name, i+1, r_auc_roc[i], r_auc_pr[i], r_time[i]))

    txt = data_name + ", AUC-ROC, %.4f, %.4f , AUC-PR, %.4f, %.4f, %.4fs, para, (%d*%d, %.4f)" % \
          (np.average(r_auc_roc), np.std(r_auc_roc), np.average(r_auc_pr), np.std(r_auc_pr), np.average(r_time),
           batch_size, n_epoch, lr)
    print(txt)
    doc = open('out.txt', 'a')
    print(txt, file=doc)
    doc.close()
    return


if __name__ == '__main__':
    input_root = "data/mammography.csv"
    runs = 10
    rand_seed = -1
    default_test_p = 0.4
    default_r = 0.1
    r_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    N_I2 = 10
    if os.path.isdir(input_root):
        for file_name in os.listdir(input_root):
            if file_name.endswith(".csv"):
                input_path = os.path.join(input_root, file_name)
                name = input_path.split("/")[-1].split('.')[0]
                this_batch_size, this_n_epoch, this_lr = config.get_run_config(name)
                main(input_path, ratio=default_r, n_run=runs, n_selected_inl=N_I2,
                     batch_size=this_batch_size, n_epoch=this_n_epoch, lr=this_lr,
                     seed=rand_seed, test_percentage=default_test_p)
                # for r in r_list:
                #     main(input_path, ratio=r, n_run=runs, n_selected_inl=N_I2,
                #          batch_size=this_batch_size, n_epoch=this_n_epoch, lr=this_lr,
                #          seed=rand_seed, test_percentage=default_test_p)

    else:
        input_path = input_root
        name = input_path.split("/")[-1].split(".")[0]
        this_batch_size, this_n_epoch, this_lr = config.get_run_config(name)
        main(input_path, ratio=default_r, n_run=runs, n_selected_inl=N_I2,
             batch_size=this_batch_size, n_epoch=this_n_epoch, lr=this_lr,
             seed=rand_seed, test_percentage=default_test_p)
