import pandas as pd
from sklearn import model_selection
from neon.data import gene
from neon.callbacks.callbacks import Callbacks
from neon.initializers import Gaussian, Xavier, Constant
from neon.layers import GeneralizedCost, Affine, Dropout
from neon.models import Model
from neon.optimizers import GradientDescentMomentum
from neon.transforms import Rectlin, Misclassification, MeanSquared, Softmax
from neon import logger as neon_logger
from neon.util.argparser import NeonArgparser
from neon.data.dataiterator import ArrayIterator, NervanaDataIterator
from neon.backends import gen_backend
import numpy as np
import sys, math
import warnings
warnings.filterwarnings('ignore')

def evaluate(model, val_iter, Metric):

    running_error = np.zeros((len(Metric.metric_names)), dtype=np.float32)
    nprocessed = 0
    dataset = val_iter
    dataset.reset()
    if hasattr(dataset, 'seq_length'):
        ndata = dataset.ndata*dataset.seq_length
    else:
        ndata = dataset.ndata
    Metric=Misclassification()
    N = 0
    for x, t in dataset:
        x = model.fprop(x, inference=True)
        # This logic is for handling partial batch sizes at the end of the dataset
        nsteps = x.shape[1] // model.be.bsz if not isinstance(x, list) else \
                 x[0].shape[1] // model.be.bsz
        bsz = min(ndata - nprocessed, model.be.bsz)

        tmp = float(running_error)
        if not math.isnan(float(running_error)):
            running_error += Metric(x, t, calcrange=slice(0, nsteps * bsz)) * nsteps * bsz
            nprocessed += bsz * nsteps

        if not math.isnan(float(running_error)): 
            running_error /= nprocessed
        if math.isnan(float(running_error)):
            running_error = tmp
            break
    neon_logger.display('Misclassification error = %.1f%%' % (running_error * 100))

if __name__ == "__main__":

    parser = NeonArgparser(__doc__)
    args = parser.parse_args()

    be = gen_backend(backend=args.backend,
                     batch_size=args.batch_size,
                     rng_seed=None,
                     device_id=args.device_id,
                     datatype=args.datatype)

    # Change the path accordingly ...locating csv file
    ip_file_path = "../../examples/gene"

    # Change the Filename accordingly
    filename = "splice_data_CAGT.csv"

    names = ['class', 'C', 'A', 'G', 'T']

    # Number of classes EI, IE and N
    nclass = 3

    # 20% of data used for validation
    validation_size = 0.20

    train_data, valid_data, train_label, valid_label = gene.load_data(ip_file_path, filename, names, validation_size = 0.20)

    train_iter = ArrayIterator(train_data, train_label, nclass=nclass, lshape=(1, 4), name='train')
    val_iter = ArrayIterator(valid_data, valid_label, nclass=nclass, lshape=(1, 4), name='valid')

    # weight
    w = Xavier()

    # bias
    b = Constant()

    # setup model layers
    # fc1, Relu, dropout
    # fc2, Relu, dropout
    # fc3, Softmax, dropout
    layers = [Affine(nout=50, init=w, bias=b, activation=Rectlin()),
              Dropout(keep=0.5),
              Affine(nout=50, init=w, bias=b, activation=Rectlin()),
              Dropout(keep=0.4),
              Affine(nout=3, init=w, bias=b, activation=Softmax()),
              Dropout(keep=0.3)
            ]

    # Optimizer
    optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9, stochastic_round=args.rounding)

    # Cost
    cost = GeneralizedCost(costfunc=MeanSquared())

    model = Model(layers=layers)

    callbacks = Callbacks(model, eval_set=val_iter, **args.callback_args)

    # Training
    model.fit(train_iter,  optimizer=optimizer, num_epochs=1, cost=cost, callbacks=callbacks)

    # Evluate
    evaluate(model, val_iter, Metric=Misclassification())


