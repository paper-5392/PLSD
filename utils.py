import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import metrics


def split_train_test(X, y, test_size, seed):
    idx_norm = y == 0
    idx_out = y == 1

    # keep outlier ratio, norm is normal out is outlier
    if seed == -1:
        rs = None
    else:
        rs = seed
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X[idx_norm], y[idx_norm],
                                                                            test_size=test_size,
                                                                            random_state=rs)
    X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(X[idx_out], y[idx_out],
                                                                        test_size=test_size,
                                                                        random_state=rs)
    X_train = np.concatenate((X_train_norm, X_train_out))
    X_test = np.concatenate((X_test_norm, X_test_out))
    y_train = np.concatenate((y_train_norm, y_train_out))
    y_test = np.concatenate((y_test_norm, y_test_out))

    # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
    scaler = StandardScaler().fit(X_train)
    X_train_stand = scaler.transform(X_train)
    X_test_stand = scaler.transform(X_test)

    # Scale to range [0,1]
    minmax_scaler = MinMaxScaler().fit(X_train_stand)
    X_train_scaled = minmax_scaler.transform(X_train_stand)
    X_test_scaled = minmax_scaler.transform(X_test_stand)

    return X_train_scaled, y_train, X_test_scaled, y_test


def semi_setting(x_train, y_train, ratio_known_outliers, seed):
    outlier_indices = np.where(y_train == 1)[0]
    inlier_indices = np.where(y_train == 0)[0]
    n_outliers = len(outlier_indices)
    n_inliers = len(inlier_indices)

    if seed == -1:
        rng = np.random.RandomState(None)
    else:
        rng = np.random.RandomState(seed)

    n_known_outliers = int(n_outliers * ratio_known_outliers)
    known_idx = rng.choice(outlier_indices, n_known_outliers, replace=False)
    new_y_train = np.zeros(x_train.shape[0], dtype=int)
    new_y_train[known_idx] = 1
    return new_y_train


def get_sorted_index(score, order='descending'):
    '''
    :param score:
    :return: index of sorted item in descending order
    e.g. [8,3,4,9] return [3,0,2,1]
    '''
    score_map = []
    size = len(score)
    for i in range(size):
        score_map.append({'index':i, 'score':score[i]})
    if order == "descending":
        reverse = True
    elif order == "ascending":
        reverse = False
    score_map.sort(key=lambda x: x['score'], reverse=reverse)
    keys = [x['index'] for x in score_map]
    return keys


def get_rank(score):
    '''
    :param score:
    :return:
    e.g. input: [0.8, 0.4, 0.6] return [0, 2, 1]
    '''
    sort = np.argsort(score)
    size = score.shape[0]
    rank = np.zeros(size)
    for i in range(size):
        rank[sort[i]] = size - i - 1

    return rank


def min_max_norm(array):
    array = np.array(array)
    _min_, _max_ = np.min(array), np.max(array)
    norm_array = np.array([(a - _min_) / (_max_ - _min_) for a in array])
    return norm_array


def get_performance(score, y_true):
    auc_roc = metrics.roc_auc_score(y_true, score)
    precision, recall, _ = metrics.precision_recall_curve(y_true, score)
    auc_pr = metrics.auc(recall, precision)
    return auc_roc, auc_pr


def ensemble_scores(score1, score2):
    '''
    :param score1:
    :param score2:
    :return: ensemble score
    @@ ensemble two score functions
    we use a non-parameter way to dynamically get the tradeoff between two estimated scores.
    It is much more important if one score function evaluate a object with high outlier socre,
    which should be paid more attention on these scoring results.
    instead of using simple average, median or other statistics
    '''

    objects_num = len(score1)

    [_max, _min] = [np.max(score1), np.min(score1)]
    score1 = (score1 - _min) / (_max - _min)
    [_max, _min] = [np.max(score2), np.min(score2)]
    score2 = (score2 - _min) / (_max - _min)

    rank1 = get_rank(score1)
    rank2 = get_rank(score2)

    alpha_list = (1. / (2 * (objects_num - 1))) * (rank2 - rank1) + 0.5
    combine_score = alpha_list * score1 + (1. - alpha_list) * score2
    return combine_score

