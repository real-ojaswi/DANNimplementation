from DANNModel import Model
import numpy as np
from sklearn.metrics import accuracy_score

class test():

    def __init__(self, model):
        self.model=model


    def test(self, x_source_sample, y_source_sample, x_target_sample=None, y_target_sample=None):
        y_pred_test_source = self.model.predict_label(x_source_sample)
        accuracy_source = accuracy_score(np.argmax(y_pred_test_source, axis=1), y_source_sample)
        if x_target_sample is not None and y_target_sample is not None:
            y_pred_test_target = self.model.predict_label(x_target_sample)
            accuracy_target = accuracy_score(np.argmax(y_pred_test_target, axis=1), y_target_sample)
            accuracy_log = {'accuracy_score_source': accuracy_source,
                            'accuracy_score_target': accuracy_target}
        else:
            accuracy_log = {'accuracy_score_source': accuracy_source}

        return accuracy_log

