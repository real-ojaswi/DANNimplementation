import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class GradientReverseLayer(Layer):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=False):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def call(self, inputs):
        coeff = 2.0 * (self.hi - self.lo) / (1.0 + tf.exp(-self.alpha * tf.cast(self.iter_num, dtype=tf.float32) / self.max_iters))- (self.hi - self.lo) + self.lo

        if self.auto_step:
            self.step()
        return inputs * coeff

    def step(self):
        self.iter_num += 1

class WarmStartGradientReverseLayer(GradientReverseLayer):
    def __init__(self, alpha=1.0, lo=0.0, hi=1.0, max_iters=1000, auto_step=False):
        super(WarmStartGradientReverseLayer, self).__init__(alpha, lo, hi, max_iters, auto_step)
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def call(self, inputs):
        coeff = 2.0 * (self.hi - self.lo) / (1.0 + tf.exp(-self.alpha * tf.cast(self.iter_num, dtype=tf.float32) / self.max_iters))- (self.hi - self.lo) + self.lo

        if self.auto_step:
            self.step()
        return inputs * coeff

    def step(self):
        self.iter_num += 1

