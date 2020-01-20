import numpy as np
import utils
import classification
import clustering
import math
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from Net import MLPDrop


class PLSD:
    def __init__(self, device, name, seed):
        self.dataset_name = name
        self.x_train = None
        self.dimension = 0
        self.device = device
        if seed == -1:
            self.seed = None
        else:
            self.seed = seed

        self.known_anomaly_indices = None
        self.inlier_indices = None
        self.init_anomaly_score = None

        self.surrogate_x = None
        self.surrogate_y = None

        self.net = None
        self.inlier_efficacy = None

    def fit(self, x_train, y_train, n_clusters=10, batch_size=64, n_epochs=30, lr=0.1):
        self.x_train = x_train
        self.dimension = x_train.shape[1]
        self.known_anomaly_indices = np.where(y_train == 1)[0]

        # identify reliable and diversified inliers
        self.inlier_indices, self.init_anomaly_score = self.explore_inliers()
        # supplement missed typical inliers
        new_inlier_indices = self.supplement_inliers(n_clusters=n_clusters)
        self.inlier_indices = np.concatenate([self.inlier_indices, new_inlier_indices])

        # surrogate supervision-based deviation learning
        self.surrogate_x, self.surrogate_y, combine_name = self.generate_surrogate_supervision()
        n_class = len(np.unique(self.surrogate_y))
        n_hidden1 = int(2 * self.dimension)
        n_feature = self.surrogate_x.shape[1]
        net = MLPDrop(n_feature=n_feature, n_hidden=n_hidden1, n_output=n_class).to(self.device)
        print(net)
        self.net, self.inlier_efficacy = classification.train(
            x=self.surrogate_x, y=self.surrogate_y,
            labeled_ano_indices=self.known_anomaly_indices, explored_inl_indices=self.inlier_indices,
            name=combine_name, net=net, device=self.device,
            batch_size=batch_size, n_epochs=n_epochs, lr=lr
        )

        print('Finished Training')
        return

    def predict(self, x_test, y_test, n_selected_inl):
        n_test = len(x_test)

        # select high-efficacy inliers
        chosen = utils.get_sorted_index(self.inlier_efficacy)[:n_selected_inl]
        inlier_indices = self.inlier_indices[chosen]
        inlier = self.x_train[inlier_indices]
        n_inlier = len(inlier_indices)

        # combine with selected high-efficacy inliers and feed into network
        combined = np.zeros([n_test*n_inlier, self.dimension*2])
        for ii, x in enumerate(x_test):
            for jj, normal in enumerate(inlier):
                combined[ii * n_inlier + jj] = np.concatenate((x, normal), axis=0)
        _, predicted_prob = classification.predict(combined, self.net, self.device)

        weight = self.inlier_efficacy[chosen]
        anomaly_score = np.zeros(n_test)
        for i in range(n_test):
            this_prob = predicted_prob[i*n_inlier: (i+1)*n_inlier]
            outlying_score = (this_prob[:, 1] - this_prob[:, 0]) * weight
            anomaly_score[i] = np.average(outlying_score)

        print("Finished Testing")
        return anomaly_score

    def explore_inliers(self):
        x = self.x_train
        N = len(x)
        known_anomaly_indices = self.known_anomaly_indices
        n_known_anomaly = len(known_anomaly_indices)

        # # iForest and HBOS are integrated to yield initial anomaly score
        clf1 = IForest(behaviour="new", random_state=self.seed)
        clf1.fit(x)
        scores1 = utils.min_max_norm(clf1.decision_scores_)
        rank1 = np.argsort(scores1) + 1
        w1 = np.average(rank1[known_anomaly_indices])

        clf2 = HBOS()
        clf2.fit(x)
        scores2 = utils.min_max_norm(clf2.decision_scores_)
        rank2 = np.argsort(scores2) + 1
        w2 = np.average(rank2[known_anomaly_indices])

        w1, w2 = w1/(w1+w2), w2/(w1+w2)
        scores1, scores2 = scores1*w1, scores2*w2
        init_anomaly_score = utils.ensemble_scores(scores1, scores2)
        init_anomaly_score = utils.min_max_norm(init_anomaly_score)

        n_sampling = max(min(200, 10*n_known_anomaly), n_known_anomaly)
        candidate_indices = utils.get_sorted_index(init_anomaly_score)[-int(0.5 * N):]
        candidate_indices = np.delete(candidate_indices, self.known_anomaly_indices)
        anomaly_score = init_anomaly_score[candidate_indices]
        sampling_prob = (1-anomaly_score) / np.sum(1-anomaly_score)
        rng = np.random.RandomState(seed=self.seed)
        inlier_indices = rng.choice(candidate_indices, n_sampling, p=sampling_prob, replace=False)

        return inlier_indices, init_anomaly_score

    def supplement_inliers(self, n_clusters=10):
        x = self.x_train
        n_x = x.shape[0]
        cluster_info, _ = clustering.do_mb_kmeans(x, n_clusters=n_clusters, batch=64, seed=self.seed)

        # set flags of clusters with known anomalies as 1
        cluster_flag = -1 * np.ones(n_clusters, dtype=int)
        for anomaly_index in self.known_anomaly_indices:
            this_cluster = cluster_info[anomaly_index]
            cluster_flag[this_cluster] = 1

        # set flags of clusters with known inliers as 0
        for inlier_index in self.inlier_indices:
            this_cluster = cluster_info[inlier_index]
            cluster_flag[this_cluster] = 0

        # avg_cluster_score = np.zeros(n_clusters)
        # cluster_count = np.zeros(n_clusters, dtype=int)
        # for i in range(n_x):
        #     this_cluster = cluster_info[i]
        #     avg_cluster_score[this_cluster] += self.init_anomaly_score[i]
        #     cluster_count[this_cluster] += 1
        # avg_cluster_score = avg_cluster_score / cluster_count
        # threshold = np.median(self.init_anomaly_score[self.known_anomaly_indices])

        # enlarge known inliers
        chosen_inlier_indices = []
        uncovered_cluster_indices = np.where(cluster_flag == -1)[0]
        for this_cluster in uncovered_cluster_indices:
            this_candidate_indices = np.where(cluster_info == this_cluster)[0]
            this_score = self.init_anomaly_score[this_candidate_indices]
            if np.sum(1-this_score) == 0:
                continue
            this_sampling_prob = (1-this_score) / np.sum(1-this_score)
            this_sampling_num = math.ceil(0.1 * len(this_candidate_indices))
            rng = np.random.RandomState(seed=self.seed)
            this_chosen_indices = rng.choice(this_candidate_indices, this_sampling_num,
                                                   p=this_sampling_prob, replace=False)
            chosen_inlier_indices.extend(this_chosen_indices)

        chosen_inlier_indices = np.array(chosen_inlier_indices, dtype=int)
        return chosen_inlier_indices

    def generate_surrogate_supervision(self):
        x = self.x_train
        ano_indices, nor_indices = self.known_anomaly_indices, self.inlier_indices

        # labeled x and y are labeled anomalies and explored inliers
        labeled_x = np.concatenate((x[ano_indices], x[nor_indices]), axis=0)
        labeled_y = np.append(np.ones(len(ano_indices), dtype=int), np.zeros(len(nor_indices), dtype=int))

        labeled_size = len(nor_indices) + len(ano_indices)
        print("labeled size: %d (%d normal + %d anomaly)" %
              (labeled_size, len(nor_indices), len(ano_indices)))
        surrogate_supervision_size = int((labeled_size * (labeled_size-1)) * 0.5)
        # relation_train_size = int((labeled_size * (labeled_size-1)) * 0.5)

        surrogate_x = np.zeros([surrogate_supervision_size, x.shape[1] * 2])
        surrogate_y = np.zeros([surrogate_supervision_size], dtype=int)
        name = np.zeros([surrogate_supervision_size, 2], dtype=int)
        count = 0
        for i in range(labeled_size):
            for j in range(i+1, labeled_size):
                surrogate_x[count] = np.append(labeled_x[i], labeled_x[j])
                surrogate_y[count] = labeled_y[i] + labeled_y[j]
                name[count] = np.array([i, j])
                count += 1

        # remove anomaly-anomaly combination
        remove_indices1 = np.where(surrogate_y == 2)[0]
        surrogate_x = np.delete(surrogate_x, remove_indices1, axis=0)
        surrogate_y = np.delete(surrogate_y, remove_indices1)
        surrogate_name = np.delete(name, remove_indices1, axis=0)

        return surrogate_x, surrogate_y, surrogate_name
