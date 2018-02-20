from itertools import combinations
import numpy as np
from gbrank import GBrank


def show_order(comments, ys, qid_lst):
    # ysに従った順序関係を示す
    qid_distinct = np.unique(qid_lst)
    for qid in qid_distinct:
        order_strs = []
        comments_in_qid = comments[qid_lst == qid]
        ys_in_qid = ys[qid_lst == qid]
        for left, right in combinations(zip(comments_in_qid, ys_in_qid), 2):
            if left[1] > right[1]:
                order_strs.append("{} > {}".format(left[0], right[0]))
            elif left[1] < right[1]:
                order_strs.append("{} > {}".format(right[0], left[0]))
        print(", ".join(order_strs))


if __name__ == '__main__':
    train_dat = (
        "3 qid:1 1:1 2:1 3:0 4:0.2 5:0 # 1A\n"
        "2 qid:1 1:0 2:0 3:1 4:0.1 5:1 # 1B\n"
        "1 qid:1 1:0 2:1 3:0 4:0.4 5:0 # 1C\n"
        "1 qid:1 1:0 2:0 3:1 4:0.3 5:0 # 1D\n"
        "1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2A\n"
        "2 qid:2 1:1 2:0 3:1 4:0.4 5:0 # 2B\n"
        "1 qid:2 1:0 2:0 3:1 4:0.1 5:0 # 2C\n"
        "1 qid:2 1:0 2:0 3:1 4:0.2 5:0 # 2D\n"
        "2 qid:3 1:0 2:0 3:1 4:0.1 5:1 # 3A\n"
        "3 qid:3 1:1 2:1 3:0 4:0.3 5:0 # 3B\n"
        "4 qid:3 1:1 2:0 3:0 4:0.4 5:1 # 3C\n"
        "1 qid:3 1:0 2:1 3:1 4:0.5 5:0 # 3D"
    )

    # train_dat読み込み
    X = []
    ys = []
    qid_lst = []
    comments = []
    for line in train_dat.split("\n"):
        elems = line.split(" ")
        ys.append(int(elems[0]))
        qid_lst.append(int(elems[1].split(":")[1]))
        xs = []
        for i in range(5):
            xs.append(float(elems[2 + i].split(":")[1]))
        X.append(xs)
        comments.append(elems[-1])
    X = np.array(X)
    ys = np.array(ys)
    qid_lst = np.array(qid_lst)
    comments = np.array(comments)

    n_trees = 20
    min_data_in_leaf = 2
    sampling_rate = 0.8
    shrinkage = 0.1
    np.random.seed(0)
    gbrank = GBrank(n_trees, min_data_in_leaf, sampling_rate, shrinkage)
    gbrank.fit(X, ys, qid_lst)
    ys_predict = gbrank.predict(X)

    print("correct order")
    show_order(comments, ys, qid_lst)

    print("\npredicted order")
    show_order(comments, ys_predict, qid_lst)
