import numpy as np
import pandas as pd
from DANNModel import Model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from sklearn.metrics import accuracy_score
import cv2
from DisplayLogs import display_logs
from DANNtest import test

class train(display_logs):
    def __init__(self, X_source, y_source, X_target=None, y_target=None, model=Model(), batch_size=64, epochs=20, source_only=False):
        super().__init__(X_source, y_source, X_target, y_target, model, source_only, epochs)
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.y_target = y_target
        self.source_only = source_only
        self.__call__()

    def __call__(self):
        list_avg_losses = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 2*len(self.X_source) // self.batch_size  # Adjust batch_size as needed
            for batch in range(num_batches):
                start_idx = int(batch * self.batch_size / 2)
                end_idx = int((batch + 1) * self.batch_size / 2)
                batch_inputs_s = self.X_source[start_idx:end_idx]
                batch_labels = self.y_source[start_idx:end_idx]
                # batch_inputs=tf.concat([batch_inputs_s, batch_inputs_t], axis=0)
                if self.source_only:
                    batch_loss = self.model.train_source_only(batch_inputs_s, batch_labels)
                else:
                    batch_inputs_t = self.X_target[start_idx:end_idx]
                    batch_loss = self.model.train_step(x_source=batch_inputs_s, x_target=batch_inputs_t, y=batch_labels)
                epoch_loss += batch_loss.numpy()

            # Calculate and display average loss for the epoch
            average_loss = epoch_loss / num_batches
            print(f'Epoch {epoch + 1} Loss: {np.mean(average_loss)}')
            list_avg_losses.append(average_loss)

            # generate random 10000 samples to test
            random_indices = np.random.choice(len(self.X_source), 10000, replace=False)
            x_source_sample = self.X_source[random_indices]
            y_source_sample = self.y_source[random_indices]
            if self.X_target is not None and self.y_target is not None:
                x_target_sample = self.X_target[random_indices]
                y_target_sample = self.y_target[random_indices]
                # get the test score
                tester = test(self.model)
                accuracy_log = tester.test(x_source_sample, y_source_sample, x_target_sample, y_target_sample)
                print(f'For epoch {epoch + 1}: {accuracy_log}')
            else:
                tester = test(self.model)
                accuracy_log = tester.test(x_source_sample, y_source_sample)
                print(f'For epoch {epoch + 1}: {accuracy_log}')

        return {'final loss': average_loss, 'all avg losses': list_avg_losses}


