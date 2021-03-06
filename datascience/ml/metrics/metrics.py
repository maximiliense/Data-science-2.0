from abc import ABC
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp
import math

from engine.path import output_path
from engine.tensorboard import add_scalar
from engine.flags import incorrect_io, deprecated

from sklearn.metrics import roc_curve, auc



class ValidationMetric(ABC):
    def __init__(self, final_validation=False, sort_needed=False):
        self.final_validation = final_validation
        self.sort_needed = sort_needed
        self.score = None
        self.cv_metric = False

    def metric_score(self):
        return self.score

    @deprecated(comment='switch to metric_score')
    def get_result(self):
        return self.metric_score()

    def is_better(self, score):
        raise NotImplemented()

    def __repr__(self):
        return self.__class__.__name__


class JustExportPredictions(ValidationMetric):
    def __init__(self, save_true_labels=False, final_validation=False):
        super().__init__(final_validation)

    def __call__(self, predictions, labels):
        np.save(output_path('predictions.npy'), predictions)
        return self.__str__()

    def __str__(self):
        return "Predictions saved at \"" + output_path('predictions.npy') + "\""


class ValidationAccuracy(ValidationMetric):
    def __init__(self, top_k=10, final_validation=False):
        super().__init__(final_validation, True)
        self.top_k = top_k
        self.cv_metric = True

    def __call__(self, predictions, labels):
        res = 0
        for i, pred in enumerate(predictions):
            answer = pred[0:self.top_k]
            if labels[i] in answer:
                res += 1
        self.score = res / labels.shape[0]
        add_scalar('Accuracy/top-{}'.format(self.top_k), self.score)
        return self.metric_score(), str(self)

    def is_better(self, score):
        return self.metric_score() > score

    def __str__(self):
        return 'Top-'+str(self.top_k) + ' accuracy of the model on the test set: %.4f' % self.score


class F1Score(ValidationMetric):
    def __init__(self, final_validation=False, threshold=0.):
        super().__init__(final_validation)
        self.cv_metric = True
        self.threshold = threshold

    def __call__(self, predictions, labels):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        for i, pred in enumerate(predictions):
            for j in range(predictions.shape[1]):
                if labels[i, 0, j] == 1:
                    if labels[i, 1, j] == 1 and predictions[i, j] >= self.threshold:
                        true_positive += 1
                    elif labels[i, 1, j] == 1 and predictions[i, j] < self.threshold:
                        false_negative += 1
                    elif labels[i, 1, j] == 0 and predictions[i, j] < self.threshold:
                        true_negative += 1
                    else:
                        false_positive += 1
        print("true_positive: %d, true_negative: %d, false_positive: %d, false_negative: %d" % (
        true_positive, true_negative, false_positive, false_negative))
        positive = true_positive + false_positive
        precision = float(true_positive) / positive if positive != 0 else 0
        recall = float(true_positive) / (true_positive + false_negative)
        print("precision: %f, recall: %f" % (precision, recall))
        beta = 0.5
        F1_score = (1.0 + beta ** 2) * precision * recall / ((beta ** 2) * precision + recall) if precision != 0 else 0

        self.score = F1_score

        return self.metric_score(), str(self)

    def is_better(self, score):
        return self.metric_score() > score

    def __str__(self):
        return 'F1 Metric: ' + str(self.metric_score())

class ROCAUC(ValidationMetric):
    def __init__( self, final_validation=False):
        super().__init__(final_validation)
        self.cv_metric = True

    def __call__( self, predictions, labels ):
        predictions = 1.0 / (1.0 + np.exp(-predictions))
        n_classes = predictions.shape[1]
        y = {}
        p = {}
        for i in range(n_classes):
            print("Classe: %d - %d - Npos: %d"%(i, np.sum(labels[:, 0, i]), np.sum(labels[:, 1, i])))
            y[i] = []
            p[i] = []
            for j, pred in enumerate(predictions):
                if labels[j, 0, i] != 0:
                    y[i].append(labels[j, 1, i])
                    p[i].append(predictions[j, i])
            print(len(y[i]))
            print(len(p[i]))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], thr[i] = roc_curve(y[i], p[i])
            print("Classe %d: "%i)
            print(thr[i])
            print(fpr[i])
            print(tpr[i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


        # Plot all ROC curves
        plt.figure()
        lw = 2
        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow'])
        self.score = 1.0
        for i, color in zip(range(n_classes), colors):
            self.score = roc_auc[i] if  roc_auc[i] < self.score else self.score
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw, color="green")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig("./ROC-AUC.png")

        return self.metric_score(), str(self)

    def is_better( self, score ):
        return self.metric_score() > score

    def __str__( self ):
        return 'ROC-AUC Metric: ' + str(self.metric_score())

class ValidationAccuracyMultiple(ValidationMetric):
    def __init__(self, list_top_k=(10,), final_validation=False):
        super().__init__(final_validation, True)
        self.list_top_k = list_top_k
        self.result = np.zeros(len(self.list_top_k))

    def __call__(self, predictions, labels):
        res = np.zeros(len(self.list_top_k), dtype=int)
        for i, pred in enumerate(predictions):
            for j, top_k in enumerate(self.list_top_k):
                answer = pred[0:top_k]
                if labels[i] in answer:
                    res[j] += 1
        self.result = res / labels.shape[0]
        return self.__str__()

    def __str__(self):
        list_out = []
        for i in range(len(self.list_top_k)):
            list_out.append(
                'Top-'+str(self.list_top_k[i]) + ' accuracy of the model on the test set: %.4f' % self.result[i]
            )
        return '\n'.join(list_out)


class ValidationAccuracyMultipleBySpecies(ValidationMetric):
    def __init__(self, list_top_k, final_validation=False):
        super().__init__(final_validation, True)
        self.list_top_k = list_top_k
        self.result = np.zeros(self.list_top_k)

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros((nb_labels, len(self.list_top_k)))
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            for j, k in enumerate(self.list_top_k):
                if rg <= k:
                    result_sp[labels[i], j] += 1
        count = count[keep]
        count = count[:, np.newaxis]
        result_sp = result_sp[np.array(keep), :]
        result_sp = result_sp/count
        self.result = np.sum(result_sp, 0) / count.shape[0]

        return self.__str__()

    def __str__(self):
        list_out = []
        for i in range(len(self.list_top_k)):
            list_out.append(
                'Top-' + str(
                    self.list_top_k[i]) + ' accuracy of the model on the test set by species: %.4f' % self.result[i]
            )
        return '\n'.join(list_out)


@incorrect_io(explanation='Path should be managed by the output_path style functions from the engine...')
class ValidationAccuracyRange(ValidationMetric):
    def __init__(self, root_dir, xp_name, max_top_k=100, final_validation=False):
        super().__init__(final_validation, True)
        self.root_dir = root_dir
        self.xp_name = xp_name
        self.file_name = self.root_dir+"/"+self.xp_name+"_result_range_top"+str(max_top_k)+".npy"
        self.max_top_k = max_top_k
        self.result = np.zeros(self.max_top_k)

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])
            for j in range(self.max_top_k):
                if rg <= j:
                    self.result[j] += 1
        self.result = self.result / labels.shape[0]
        np.save(self.file_name, self.result)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy range saved in file \'"+self.file_name+"\'"


class ValidationAccuracyRangeBySpecies(ValidationMetric):
    def __init__(self, max_top_k=100, final_validation=False):
        super().__init__(final_validation, True)
        self.file_name = output_path("_result_range_top"+str(max_top_k)+"_by_species.npy")
        self.max_top_k = max_top_k
        self.result = np.zeros(self.max_top_k)

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros((nb_labels, self.max_top_k))
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            for j in range(self.max_top_k):
                if rg <= j:
                    result_sp[labels[i], j] += 1
        count = count[keep]
        count = count[:, np.newaxis]
        result_sp = result_sp[np.array(keep), :]
        result_sp = result_sp/count
        self.result = np.sum(result_sp, 0) / count.shape[0]
        np.save(self.file_name, self.result)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy range by species saved in file \'"+self.file_name+"\'"


class ValidationAccuracyForAllSpecies(ValidationMetric):
    def __init__(self, train, top_k=30, n_species=4520, final_validation=False):
        super().__init__(final_validation, True)
        self.file_name = output_path("_result_top"+str(top_k)+"_for_all_species.npy")
        self.top_k = top_k
        self.train = train
        self.prior = np.zeros(n_species, dtype=int)
        for label in self.train.labels:
            self.prior[label] += 1

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        result_sp = np.zeros(nb_labels)
        count = np.zeros(nb_labels)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            rg = np.argwhere(pred == labels[i])[0, 0]
            count[labels[i]] += 1
            keep[labels[i]] = True
            if rg <= self.top_k:
                result_sp[labels[i]] += 1
        self.prior = self.prior[keep]
        count = count[keep]
        result_sp = result_sp[keep]
        result_sp = result_sp/count
        result_sp = result_sp[np.argsort(-self.prior)]
        np.save(self.file_name, result_sp)
        return self.__str__()

    def __str__(self):
        return "Results of accuracy top"+str(self.top_k)+" for all species saved in file \'"+self.file_name+"\'"


@incorrect_io(explanation='Path should be managed by the output_path style functions from the engine...')
class ValidationAccuracyRangeBySpeciesByFrequency(ValidationMetric):
    def __init__(self, root_dir, xp_name, prior, max_top_k=100, cat=(5, 20, 50, 100), final_validation=False):
        super().__init__(final_validation, True)
        self.root_dir = root_dir
        self.xp_name = xp_name
        self.dir_name = self.root_dir + "/" + self.xp_name + "_by_species_by_frequency"
        if not os.path.isdir(self.dir_name):
            os.mkdir(self.dir_name)
        self.max_top_k = max_top_k
        self.cat = cat
        self.prior = prior
        self.splitpreds = [[] for i in range(len(cat)+1)]
        self.splitlabels = [[] for i in range(len(cat) + 1)]
        self.borne = [0]
        self.borne.extend(cat)
        m = np.max(self.prior)
        self.borne.append(m)

    def __call__(self, predictions, labels):
        predictions = predictions.tolist()
        for i, pred in enumerate(predictions):
            for j in range(len(self.cat)+1):
                if j == len(self.cat):
                    self.splitpreds[len(self.cat)].append(pred)
                    self.splitlabels[len(self.cat)].append(labels[i])
                elif self.prior[labels[i]] <= self.cat[j]:
                    self.splitpreds[j].append(pred)
                    self.splitlabels[j].append(labels[i])
                    break
        for i in range(len(self.splitpreds)):
            name = "species_between_"+str(self.borne[i]+1)+"_and_"+str(self.borne[i+1])+"_occurrences"
            if len(self.splitpreds[i])>0:
                metric = ValidationAccuracyRangeBySpecies(self.dir_name, name, max_top_k=self.max_top_k)
                res = str(metric(np.asarray(self.splitpreds[i]), np.asarray(self.splitlabels[i])))
        return self.__str__()

    def __str__(self):
        return "Results of accuracy range by species by frequency saved in folder \'"+self.dir_name+"/\'"


class ValidationMRR(ValidationMetric):
    def __init__(self, val_limit=None, final_validation=False):
        super().__init__(final_validation, True)
        self.val_limit = val_limit
        self.result = 0

    def __call__(self, predictions, labels):
        if self.val_limit is not None:
            predictions = predictions[:, :self.val_limit]
        res = 0
        for i, pred in enumerate(predictions):
            pos = np.where(pred == labels[i])
            if pos[0].shape[0] != 0:
                res += 1.0 / (pos[0][0] + 1)
        self.result = res / labels.shape[0]
        return self.__str__()

    def __str__(self):
        return 'MRR of the model on the test set: %.4f' % self.result


class ValidationMRRBySpecies(ValidationMetric):
    def __init__(self, final_validation=False):
        super().__init__(final_validation, True)
        self.result = 0

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        res = np.zeros(nb_labels, dtype=float)
        count = np.zeros(nb_labels, dtype=int)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            count[labels[i]] += 1
            keep[labels[i]] = True
            res[labels[i]] += 1.0 / (np.where(pred == labels[i])[0][0] + 1)
        res = res[keep]
        count = count[keep]
        self.result = np.sum(np.divide(res, count), dtype=float) / res.shape[0]
        return self.__str__()

    def __str__(self):
        return 'MSMRR of the model on the test set: %.4f' % self.result


class ValidationMedianRank(ValidationMetric):
    def __init__(self, final_validation=False):
        super().__init__(final_validation, True)
        self.result = 0

    def __call__(self, predictions, labels):
        res = []
        for i, pred in enumerate(predictions):
            res.append(np.where(pred == labels[i])[0][0] + 1)
        self.result = np.median(res)
        return self.__str__()

    def __str__(self):
        return 'MR of the model on the test set: %.1f' % self.result


class ValidationMedianRankBySpecies(ValidationMetric):
    def __init__(self, final_validation=False):
        super().__init__(final_validation, True)
        self.result = 0

    def __call__(self, predictions, labels):
        nb_labels = predictions.shape[1]
        res = np.zeros(nb_labels, dtype=float)
        count = np.zeros(nb_labels, dtype=int)
        keep = np.zeros(nb_labels, dtype=bool)
        for i, pred in enumerate(predictions):
            count[labels[i]] += 1
            keep[labels[i]] = True
            res[labels[i]] += np.where(pred == labels[i])[0][0] + 1
        res = res[keep]
        count = count[keep]
        self.result = np.median(np.divide(res, count))
        return self.__str__()

    def __str__(self):
        return 'MSMR of the model on the test set: %.4f' % self.result


class ValidationInfoMut(ValidationMetric):
    def x_log2_x(self, x):
        if x > 0:
            return x * math.log2(x)
        return 0

    def __init__(self, final_validation=False):
        super().__init__(final_validation)
        self.result = np.zeros(3, dtype=float)
        self.h_y = 0
        self.h_pred = 0
        self.info_mut = 0

    def __call__(self, predictions, labels):
        n_labels = predictions.shape[1]
        matrix = np.zeros((n_labels, n_labels), dtype=float)
        for i, pred in enumerate(predictions):
            matrix[labels[i]] = matrix[labels[i]] + (pred / np.sum(pred))

        matrix = matrix / np.sum(matrix)

        p_pred = np.sum(matrix, axis=0, dtype=float)
        p_y = np.sum(matrix, axis=1, dtype=float)

        f = np.vectorize(self.x_log2_x)

        h_pred = -np.sum(f(p_pred), dtype=float)
        h_y = -np.sum(f(p_y), dtype=float)
        h_joint = -np.sum(f(matrix), dtype=float)
        self.h_pred, self.h_y, self.info_mut = h_pred, h_y, h_pred + h_y - h_joint
        self.result = np.asarray([self.h_y, self.h_pred, self.info_mut])
        return self.__str__()

    def __str__(self):
        result = ''
        result += 'Entropy of groundtruth: %.4f\n' % self.h_y
        result += 'Entropy of predictions: %.4f\n' % self.h_pred
        result += 'Mutual Information: %.4f%% (%.4f)' % (max((self.info_mut / self.h_y) * 100, 0), self.info_mut)
        return result


class ValidationInfoMutBySpecies(ValidationMetric):
    def x_log2_x(self, x):
        if x > 0:
            return x * math.log2(x)
        return 0

    def __init__(self, final_validation=False):
        super().__init__(final_validation)
        self.h_y = 0
        self.h_pred = 0
        self.info_mut = 0

    def __call__(self, predictions, labels):
        n_labels = predictions.shape[1]
        matrix = np.zeros((n_labels, n_labels), dtype=float)
        count = np.zeros(n_labels, dtype=int)
        for i, pred in enumerate(predictions):
            count[labels[i]] += 1
            matrix[labels[i]] = matrix[labels[i]] + (pred / np.sum(pred))

        count.reshape((n_labels, 1))
        np.divide(matrix, count, out=np.zeros_like(matrix), where=count != 0)
        matrix = matrix / n_labels

        p_pred = np.sum(matrix, axis=0, dtype=float)
        p_y = np.sum(matrix, axis=1, dtype=float)

        f = np.vectorize(self.x_log2_x)

        h_pred = -np.sum(f(p_pred), dtype=float)
        h_y = -np.sum(f(p_y), dtype=float)
        h_joint = -np.sum(f(matrix), dtype=float)
        self.h_pred, self.h_y, self.info_mut = h_pred, h_y, h_pred + h_y - h_joint
        self.result = np.asarray([self.h_y, self.h_pred, self.info_mut])
        return self.__str__()

    def __str__(self):
        result = ''
        result += 'Entropy of uniform groundtruth: %.4f\n' % self.h_y
        result += 'Entropy of predictions by species: %.4f\n' % self.h_pred
        result += 'Mutual Information by species: %.4f%% (%.4f)' % (max((self.info_mut / self.h_y) * 100, 0), self.info_mut)
        return result


def accuracy(output, target, top_k=(10,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)

    # batch_size = target.size(0)

    _, prediction = output.topk(max_k, 1, True, True)
    prediction = prediction.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    result = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        result.append(correct_k.mul_(100.0))
    return result


def mrr(output, target):
    # batch_size = target.size(0)

    _, prediction = output.topk(200, 1, True, True)
    prediction = prediction.t()

    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    numpy_prediction = correct.cpu().data.numpy()
    return sum(1. / (np.where(numpy_prediction == 1)[0] + 1), 0)


def mrr_map(output, target, index, map_mrr, map_count):
    _, prediction = output.topk(200, 1, True, True)
    prediction = prediction.t()

    correct = prediction.eq(target.view(1, -1).expand_as(prediction))

    numpy_prediction = correct.cpu().data.numpy()

    where = np.where(numpy_prediction == 1)
    mrr_score = 1. / (where[0] + 1)
    for i in range(mrr_score.shape[0]):
        c = int(index[i][0].data.cpu().numpy())
        r = int(index[i][1].data.cpu().numpy())
        map_mrr[r][c] += mrr_score[i]
        map_count[r, c] += 1
