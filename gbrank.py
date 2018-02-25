from itertools import combinations
import numpy as np
from regression_tree import RegressionTree


class GBrank(object):
    def __init__(self, n_trees, min_data_in_leaf, sampling_rate, shrinkage, tau=0.5):
        self.n_trees = n_trees
        self.min_data_in_leaf = min_data_in_leaf
        self.sampling_rate = sampling_rate
        self.shrinkage = shrinkage
        self.tau = tau
        self.trees = []

    def fit(self, X, ys, qid_lst):
        for n_tree in range(self.n_trees):
            # 最初の木は0を返すだけの木とする
            if n_tree == 0:
                # X = [[0]], ys = [0] を学習させれば0を返す木となる
                rt = RegressionTree(1)
                rt.fit(np.array([[0]]), np.array([0]))
                self.trees.append(rt)
                continue

            target_index = np.random.choice(
                X.shape[0],
                int(X.shape[0] * self.sampling_rate),
                replace=False
            )
            X_target = X[target_index]
            ys_target = ys[target_index]
            qid_target = qid_lst[target_index]

            # 直前の木々で予測を行う
            ys_predict = self._predict(X_target, n_tree)

            # 出現するqid
            qid_target_distinct = np.unique(qid_target)

            # 各qidに対応する訓練データを取得し, n_tree本目の木で学習する訓練データを生成する
            X_train_for_n_tree = []
            ys_train_for_n_tree = []
            for qid in qid_target_distinct:
                X_target_in_qid = X_target[qid_target == qid]
                ys_target_in_qid = ys_target[qid_target == qid]
                ys_predict_in_qid = ys_predict[qid_target == qid]

                for left, right in combinations(enumerate(zip(ys_target_in_qid, ys_predict_in_qid)), 2):
                    ind_1, (ys_target_1, ys_predict_1) = left
                    ind_2, (ys_target_2, ys_predict_2) = right

                    # (ys_target_1 > ys_target_2)となるように交換
                    if ys_target_1 < ys_target_2:
                        ys_target_1, ys_target_2 = ys_target_2, ys_target_1
                        ys_predict_1, ys_predict_2 = ys_predict_2, ys_predict_1
                        ind_1, ind_2 = ind_2, ind_1

                    if ys_predict_1 < ys_predict_2 + self.tau:
                        X_train_for_n_tree.append(X_target_in_qid[ind_1])
                        ys_train_for_n_tree.append(ys_target_in_qid[ind_1] + self.tau)
                        X_train_for_n_tree.append(X_target_in_qid[ind_2])
                        ys_train_for_n_tree.append(ys_target_in_qid[ind_2] - self.tau)
            X_train_for_n_tree = np.array(X_train_for_n_tree)
            ys_train_for_n_tree = np.array(ys_train_for_n_tree)

            # n_tree本目の木を学習
            rt = RegressionTree(self.min_data_in_leaf)
            rt.fit(X_train_for_n_tree, ys_train_for_n_tree)
            self.trees.append(rt)

    def _predict(self, X, n_predict_trees):
        # n_predict_trees本の木による予測結果リストを求める
        predict_lst_by_trees = [self.trees[n_tree].predict(X) for n_tree in range(n_predict_trees)]
        # 各木による予測を統合する
        predict_result = predict_lst_by_trees[0]
        for n_tree in range(1, n_predict_trees):
            predict_result = (n_tree * predict_result + self.shrinkage * predict_lst_by_trees[n_tree]) / (n_tree + 1)
        return predict_result

    def predict(self, X):
        return self._predict(X, len(self.trees))
