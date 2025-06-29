# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: PAVITHRA K

*INTERN ID*: CT04DF2827

*DOMAIN*: DATA SCIENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH

##DESCRIPTION:In this project, I developed an image classification model using deep learning techniques implemented with Python and TensorFlow. The primary objective was to create a convolutional neural network (CNN) capable of accurately classifying images into ten categories from the widely used CIFAR-10 dataset. This project demonstrates the complete workflow of designing, training, evaluating, and visualizing the performance of a neural network for supervised learning tasks.

The project began by loading the CIFAR-10 dataset, which is conveniently included in TensorFlow’s datasets module. This dataset consists of 60,000 32×32 color images evenly distributed across ten different classes, such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. The dataset is split into 50,000 training examples and 10,000 test examples. Each image is represented as a three-dimensional array corresponding to its height, width, and color channels.

Before training, all pixel values were normalized to a range between 0 and 1 by dividing each value by 255.0. Normalization is an essential preprocessing step in deep learning, as it helps accelerate convergence and improve numerical stability during training. Additionally, a data augmentation pipeline was implemented using TensorFlow’s Keras preprocessing layers. This augmentation included random horizontal flipping, rotation, and zooming. These techniques increase the diversity of the training data and help prevent overfitting by exposing the model to more varied inputs.

The CNN architecture was designed as a sequential model, consisting of multiple convolutional and pooling layers. The first convolutional layer applies 32 filters with a 3×3 kernel size and ReLU activation. This is followed by a max pooling layer to reduce spatial dimensions while retaining essential features. The network then adds deeper convolutional layers with 64 filters to learn increasingly complex representations of the input images. After the convolutional stack, the output is flattened and passed to a dense layer with 128 neurons and ReLU activation. To further combat overfitting, a dropout layer with a rate of 0.5 randomly deactivates half of the neurons during training. The final output layer is a dense layer with ten units corresponding to the ten possible classes.

The model was compiled using the Adam optimizer, which provides an adaptive learning rate for faster convergence. The loss function selected was sparse categorical crossentropy, appropriate for multi-class classification tasks where labels are provided as integers. Model performance was evaluated using accuracy as the primary metric.

Training proceeded over 20 epochs, with both training and validation accuracy monitored at each step. After training completed, the model achieved approximately 67% test accuracy, demonstrating its capacity to learn meaningful features from the CIFAR-10 dataset.

In addition to numerical metrics, visualizations were created to inspect the model’s predictions. Matplotlib was used to display sample test images alongside their predicted labels and true labels. Correct predictions were highlighted in green, while incorrect predictions appeared in red, making it easy to identify performance strengths and weaknesses.

All development work was carried out in the PyCharm IDE, leveraging Python 3.10 and TensorFlow 2.x. This project illustrates a complete, reproducible deep learning workflow, from data loading and augmentation to model building, training, evaluation, and visualization.

