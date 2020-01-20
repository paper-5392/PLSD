import torch
import torch.utils.data as Data
import numpy as np
import time
import utils


def train(x, y, labeled_ano_indices, explored_inl_indices, name, net, device, batch_size=64, n_epochs=50, lr=0.1):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    x = torch.tensor(x, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.int64).to(device)
    train_dataset = Data.TensorDataset(x, y)
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = net(batch_x)
            loss = loss_func(out, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_train_time = time.time() - epoch_start_time
        if (epoch+1) % 5 == 0:
            print(f'| Epoch: {epoch + 1:03}/{n_epochs:03} | Train Time: {epoch_train_time:.3f}s' 
                  f'| Train Loss: {epoch_loss / n_batches:.6f} |')

    with torch.no_grad():
        out = net(x)
        score = out.cpu().data.numpy()
    efficacy = np.zeros(len(labeled_ano_indices) + len(explored_inl_indices))

    label1_indices = np.where(y.cpu() == 1)[0]
    for ind in label1_indices:
        this_score = score[ind]
        nor_index = name[ind][1]
        efficacy[nor_index] += this_score[1] - this_score[0]
    inl_efficacy = utils.min_max_norm(efficacy[-len(explored_inl_indices):])
    return net, inl_efficacy


def predict(x, net, device):
    x = torch.tensor(x, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = net(x)
        out = out.cpu()
        _, predict = torch.max(out.data, 1)
        predict = predict.numpy()
        score = out.data.numpy()
    return predict, score


def test(x, y, net, batch_size=64):
    n_class = len(np.unique(y))
    x_test = torch.tensor(x, dtype=torch.float32)
    y_test = torch.tensor(y, dtype=torch.int64)
    test_dataset = Data.TensorDataset(x_test, y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    correct = 0
    correct_list = np.zeros(n_class)
    predict = []
    ground_truth = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            output = net(batch_x)
            _, batch_predict = torch.max(output.data, 1)
            correct += (batch_predict == batch_y).sum().item()

            predict.extend(batch_predict.numpy().tolist())
            ground_truth.extend(batch_y.numpy().tolist())

            mark = (batch_predict == batch_y).numpy()
            for i in range(n_class):
                correct_list[i] += len(np.intersect1d(np.where(mark == 1)[0], np.where(batch_y == i)[0]))

    predict = np.array(predict)

    class_total = np.zeros(n_class)
    predict_total = np.zeros(n_class)
    for i in range(n_class):
        class_total[i] = len(np.where(y == i)[0])
        predict_total[i] = len(np.where(predict == i)[0])

    for i in range(n_class):
        print("Ground Truth: class%d: %d" % (i, class_total[i]))
    for i in range(n_class):
        print("Predict: class%d: %d" % (i, predict_total[i]))

    print('Total Acc: %.2f %%' % (100 * correct / len(y)))
    for i in range(n_class):
        print("Class %d Acc, %.2f%%" % (i,  100 * correct_list[i] / class_total[i]))

