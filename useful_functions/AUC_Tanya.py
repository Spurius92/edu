# -*- coding: utf-8 -*-

"""
An example to check the AUC score on a validation set for each 10 epochs.
I hope it will be helpful for optimizing number of epochs.
"""
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback


class IntervalEvaluation(Callback):
    """
    calculates AUC for train and val sets and passes them as lists every n epochs
    """
    def __init__(self, train_data =(), validation_data=(), interval=10):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_train, self.y_train = train_data
        self.X_val, self.y_val = validation_data
        self.auc = []
        self.auc_train = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            #auc for traibn data
            y_pred_train = self.model.predict(self.X_train, verbose=0)
            score_train = roc_auc_score(self.y_train, y_pred_train)
            self.auc_train.append(score_train)
            #auc for validation data
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            self.auc.append(score)
            print("epoch: {:d} - roc_auc_score train: {:.6f}".format(epoch, score_train))
            print("epoch: {:d} - roc_auc_score val: {:.6f}".format(epoch, score))
            #logging.info("epoch: {:d} - score: {:.6f}".format(epoch, score))

# (snip)
"""
if __name__ == '__main__':
    l.basicConfig(format='%(asctime)s %(message)s', level=l.INFO)
    X_train, y_train, X_test, y_test = load_data()
    ival = IntervalEvaluation(validation_data=(X_test, y_test), interval=10)

    clf = keras_model(input_size=X_train.shape[1])
    clf.fit(X_train, y_train, nb_epoch=100, batch_size=128, verbose=0, callbacks=[ival])
    
"""   