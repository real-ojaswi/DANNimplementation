# Domain-Adversarial Training of Neural Networks (DANN) Implementation

This repository contains the implementation of Domain-Adversarial Training of Neural Networks (DANN) using TensorFlow. The `DANN.ipynb` notebook demonstrates experiments on the MNIST and USPS datasets. The model is trained on MNIST and its accuracy is evaluated on the USPS dataset both with and without domain adaptation techniques.

## Components

- **DANNModel.py**: Contains the implementation of the DANN model.
- **grl.py**: Implements the Gradient Reversal Layer, adapted from the [DALN repository](https://github.com/xiaoachen98/DALN) originally written for PyTorch.
- **DALNtrain.py**: Provides training utilities for the DANN model.
- **DisplayLogs.py**: Contains utilities for displaying and calculating accuracies during training.
- **DANN.ipynb**: Jupyter notebook for conducting experiments and showcasing results.

## Results

The experimental results show a significant increase in accuracy when using DANN compared to training without domain adaptation:

| Method       | Source Accuracy | Target Accuracy |
|--------------|-----------------|-----------------|
| Source only  | 0.9998          | 0.417           |
| DANN         | 0.9341          | 0.713           |

## Training Procedure

To train the model:

1. **Import Required Modules**: Import TensorFlow and load the MNIST dataset, resizing images to 32x32x3 for compatibility.
   
2. **Initialize Model**: Import the model from `DANNModel.py` and initialize it.

3. **Training Setup**: Import `train` from `DALNtrain.py` and initialize a training object (`trainer`) with parameters including `X_source`, `y_source`, `model`, `batch_size`, `X_target`, `y_target`, `epochs`, and `source_only` boolean flag.

4. **Run Training**: Set `source_only=True` to train without domain adaptation, or `source_only=False` to train with domain adaptation.

5. **Prediction**: Use the `predict_label` method of the model object to predict labels.

## Calculating Accuracy

- During training, source and target accuracies are displayed.
- Alternatively, use the `accuracy_score` function from scikit-learn or import `display_logs` from `DisplayLogs.py` for automated accuracy calculation and logging.

## Originality of Code

- The code in `grl.py` was adapted from the DALN repository originally written in PyTorch.
- The architecture of the model has been simplified for computational efficiency while retaining effectiveness.

## Datasets

- **MNIST**: Handwritten digits dataset imported from `tensorflow.keras.datasets.mnist`.
- **USPS**: Handwritten digits dataset imported from `extra_keras.usps`.

Both datasets are small-sized with sufficient samples, making them suitable for this domain adaptation experiment due to their notable differences.

## References

- [DALN Repository](https://github.com/xiaoachen98/DALN)
- [Original Paper on Domain-Adversarial Neural Networks](https://arxiv.org/pdf/1505.07818.pdf)

Feel free to explore and contribute to this repository to enhance domain adaptation techniques using DANN.
