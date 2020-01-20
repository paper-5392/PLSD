
def get_run_config(data_name):
    n_epoch = 30
    batch_size = 64
    lr = 0.1

    if data_name in ["fraud", "spambase"]:
        n_epoch = 30
        batch_size = 256

    if data_name in ["annthyroid"]:
        n_epoch = 100

    if data_name in ["ad"]:
        lr = 0.001

    return batch_size, n_epoch, lr
