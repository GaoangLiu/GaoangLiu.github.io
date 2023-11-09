import sklearn
import sklearn.metrics
import collections

def _calculate(y_true, y_pred):
    tp = collections.defaultdict(int)  # True Positive
    fp = collections.defaultdict(int)  # False Positive
    fn = collections.defaultdict(int)  # False Negative
    for l, p in zip(y_true, y_pred):
        if l == p:
            tp[p] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    return tp, fp, fn


def precision_recall(y_true, y_pred):
    tp, fp, fn = _calculate(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    print("{:>10} {:>10} {:>10} {:>10}".format(
        '', 'precision', 'recall', 'f1_score'))
    for l in labels:
        # Be careful, tp[x], fp[x] can be zero at the same time.
        prec = 0 if tp[l] == 0 else tp[l] / (tp[l] + fp[l])  # precision
        rec = tp[l] / (tp[l] + fn[l])                        # recall
        f1 = 0 if prec * rec == 0 else 2 * (prec * rec) / (prec + rec)
        print("{:>10} {:>10.4f} {:>10.4f} {:>10.4f}".format(l, prec, rec, f1))


def micro_f1_score(y_true, y_pred):
    tp, fp, fn = _calculate(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    c_tp = sum(tp.values())
    c_fp = sum(fp.values())
    c_fn = sum(fn.values())
    prec = 0 if c_tp == 0 else c_tp / (c_tp + c_fp)
    rec = c_tp / (c_tp + c_fn)
    print("f1_score micro ", 0 if prec * rec ==
          0 else 2 * (prec * rec) / (prec + rec))


def weighted_f1_score(y_true, y_pred):
    tp, fp, fn = _calculate(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    cnt = collections.Counter(y_true)
    weighted_score = 0
    for l in labels:
        prec = 0 if tp[l] == 0 else tp[l] / (tp[l] + fp[l])  # precision
        rec = tp[l] / (tp[l] + fn[l])                       # recall
        f1 = 0 if prec * rec == 0 else 2 * (prec * rec) / (prec + rec)
        weighted_score += f1 * cnt[l] / len(y_true)
    print("f1_score_weighted", weighted_score)


y_true = [0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 0, 2, 1, 0, 1, 1, 0]
print(sklearn.metrics.classification_report(y_true, y_pred))

precision_recall(y_true, y_pred)
micro_f1_score(y_true, y_pred)
print("micro average", sklearn.metrics.f1_score(y_true, y_pred, average='micro'))

weighted_f1_score(y_true, y_pred)
print("weighted average", sklearn.metrics.f1_score(y_true, y_pred, average='weighted'))
