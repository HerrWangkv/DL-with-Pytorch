from IPython import display
from matplotlib import pyplot as plt
import random
import torch

def use_svg_display():
    display.set_matplotlib_formats('svg') # display in "svg" format

def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()    
    plt.rcParams["figure.figsize"] = figsize

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices) 
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)]) # datatype of the element is `long`
        yield features.index_select(0, j), labels.index_select(0, j)
    
def linreg(X, w, b):
    assert X.shape[1] == w.shape[0]
    assert w.shape[1] == b.shape[0]
    return torch.mm(X, w) + b # matrix multiplication

def squared_loss(y_hat, y):
    # return a loss vector with the same shape as y_hat
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        # only change value and will not be tracked
        param.data -= lr * param.grad / batch_size
    

