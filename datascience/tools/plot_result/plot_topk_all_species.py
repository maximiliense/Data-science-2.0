import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression


class ModelPrint(object):
    def __init__(self, name, path, color, linestyle):
        self.name = name
        self.path = path
        self.color = color
        self.linestyle = linestyle
        self.list_curves = []


def cumul_mean(arr):
    progressive = np.ndarray(arr.shape[0])
    sum = 0
    for i, score in enumerate(arr):
        sum += score
        progressive[i] = sum / (i + 1)
    return progressive


def mean_categorical(arr, k=200):
    size = (int)(arr.shape[0]/k)+1
    print(size)
    cat = np.ndarray(size)
    grad = np.arange(size)
    for i in range(size):
        if i == size-1:
            cat[i] = np.mean(arr[i*k:])
        else:
            cat[i] = np.mean(arr[i*k:(i+1)*k])
    return cat, grad


def slide_mean(arr, start=1, k=200):
    borne = int(k/2)
    res = np.ndarray(arr.shape[0])
    for i in range(arr.shape[0]):
        if start <= borne:
            res[i] = np.mean(arr[:i+start])
            start += 1
        elif i < borne:
            res[i] = np.mean(arr[:i+borne+1])
        elif i >= arr.shape[0]-borne:
            res[i] = np.mean(arr[i-borne:])
        else:
            res[i] = np.mean(arr[i-borne:i+borne+1])
    return res


def linear_regression(arr, k=5):
    coeff = [np.arange(arr.shape[0])]
    for i in range(k-1):
        coeff.append(coeff[0]**(i+2))
    X = np.stack(coeff).T
    reg = LinearRegression().fit(X, arr)
    res = np.dot(X, reg.coef_) + reg.intercept_
    return res

bt = np.load("/home/bdeneu/old_computer/home/results/bt_env_rs_result_top30_for_all_species.npy")
rf = np.load("/home/bdeneu/old_computer/home/results/rf_env_rs_d16_result_top30_for_all_species.npy")
cnn = np.load("/home/bdeneu/old_computer/home/results/inception_rs_normal_do07_4_result_top30_for_all_species.npy")
dnn = np.load("/home/bdeneu/old_computer/home/results/inception_rs_constant_do05_result_top30_for_all_species.npy")


grad_bt = np.arange(bt.shape[0])
grad_rf = np.arange(rf.shape[0])
grad_cnn = np.arange(cnn.shape[0])
grad_dnn = np.arange(dnn.shape[0])

print(grad_bt)
print(grad_rf)
print(grad_cnn)
print(grad_dnn)
bt_progressive = slide_mean(bt)
rf_progressive = slide_mean(rf)
cnn_progressive = slide_mean(cnn)
dnn_progressive = slide_mean(dnn)

s = ""
s = s + ";".join([str(i) for i in grad_cnn])
s = s +"\n" + ";".join([str(i) for i in cnn_progressive])
s = s +"\n" + ";".join([str(i) for i in dnn_progressive])
s = s +"\n" + ";".join([str(i) for i in rf_progressive])
s = s +"\n" + ";".join([str(i) for i in bt_progressive])
print(s.replace(".", ","))


#plt.plot(grad, cnn_top_species, color="green", linestyle='-', label="CNN")
plt.plot(grad_dnn, dnn_progressive, linestyle="-", color="k", label="DNN")
plt.plot(grad_bt, bt_progressive, linestyle="-", color="blue", label="BT")
plt.plot(grad_rf, rf_progressive, linestyle="-", color="red", label="RF")
plt.plot(grad_cnn, cnn_progressive, linestyle="-", color="green", label="CNN")

majorLocator = MultipleLocator(5)
majorFormatter = FormatStrFormatter('%d')
minorLocator = MultipleLocator(1)

axes = plt.gca()
axes.set_xlim(0, 2359)
axes.set_ylim(0, 1.0)
axes.set_xlabel('species ordered by frequency')
axes.set_ylabel('top30 accuracy')

axes.xaxis.set_major_locator(MultipleLocator(1000))
axes.xaxis.set_minor_locator(MultipleLocator(100))

axes.yaxis.set_major_locator(MultipleLocator(0.1))
axes.yaxis.set_minor_locator(MultipleLocator(0.01))
plt.grid()
plt.legend(loc=2)

plt.show()
