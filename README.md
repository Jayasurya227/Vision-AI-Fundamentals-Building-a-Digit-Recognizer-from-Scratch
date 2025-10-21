# Digit Recognizer (Vision AI)
# Vision AI Fundamentals: Building a Digit Recognizer from Scratch ✍️

This project explores the fundamentals of Computer Vision and Deep Learning by building a Convolutional Neural Network (CNN) from scratch to recognize handwritten digits. The model is trained and evaluated on the classic MNIST dataset.

This notebook serves as a practical introduction to image classification using TensorFlow and Keras, demonstrating the key steps involved in building, training, and evaluating a deep learning model for a vision task.

**Dataset:** MNIST Handwritten Digit Dataset (provided via TensorFlow/Keras datasets API)
**Focus:** Demonstrating image data preprocessing, building a CNN architecture, training a deep learning model, evaluating performance on image classification, and visualizing results.
**Repository:** [https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch](https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch)

***

## Key Techniques & Concepts Demonstrated

Based on the analysis within the notebook (`8_Vision_AI_Fundamentals_Building_a_Digit_Recognizer_from_Scratch (2).ipynb`), the following key concepts and techniques are applied:

* **Computer Vision Fundamentals:** Understanding the task of image classification.
* **Deep Learning with TensorFlow/Keras:** Utilizing the Keras API within TensorFlow to build and train neural networks.
* **MNIST Dataset:** Working with a standard benchmark dataset for image classification.
* **Image Data Preprocessing:**
    * **Reshaping:** Adding a channel dimension to the grayscale images (e.g., 28x28 -> 28x28x1).
    * **Normalization:** Scaling pixel values from the range [0, 255] to [0, 1] to improve model training stability and performance.
* **Convolutional Neural Networks (CNNs):**
    * Building a sequential model using core CNN layers:
        * `Conv2D`: Applying convolutional filters to extract spatial hierarchies of features.
        * `MaxPooling2D`: Down-sampling feature maps to reduce dimensionality and increase robustness.
        * `Flatten`: Converting 2D feature maps into a 1D vector for the fully connected layers.
        * `Dense`: Standard fully connected neural network layers.
        * `Dropout`: Applying dropout regularization to prevent overfitting.
    * Understanding activation functions (`relu`, `softmax`).
* **Model Compilation:** Configuring the model for training using:
    * **Optimizer:** `adam` (Adaptive Moment Estimation).
    * **Loss Function:** `sparse_categorical_crossentropy` (suitable for integer-based multi-class classification labels).
    * **Metrics:** `accuracy`.
* **Model Training:** Fitting the model to the training data using `model.fit()`, specifying batch size and epochs.
* **Model Evaluation:** Assessing the trained model's performance on the unseen test dataset using `model.evaluate()` to get loss and accuracy.
* **Prediction & Visualization:**
    * Making predictions on test images using `model.predict()`.
    * Visualizing test images alongside their true and predicted labels.

***

## Analysis Workflow

The notebook follows a standard deep learning workflow for image classification:

1.  **Setup & Data Loading:** Importing necessary libraries (TensorFlow, Keras, NumPy, Matplotlib) and loading the MNIST dataset directly using `keras.datasets.mnist.load_data()`.
2.  **Data Exploration & Preprocessing:**
    * Inspecting the shape and data type of the training and testing images/labels.
    * Visualizing sample images from the dataset.
    * Reshaping the image data to include the channel dimension (28x28x1).
    * Normalizing pixel values to the range [0, 1].
3.  **CNN Model Architecture Definition:**
    * Creating a `Sequential` Keras model.
    * Adding `Conv2D`, `MaxPooling2D`, `Flatten`, `Dropout`, and `Dense` layers with appropriate configurations (number of filters, kernel size, activation functions).
4.  **Model Compilation:** Compiling the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
5.  **Model Training:** Training the compiled model on the preprocessed training data (`x_train`, `y_train`) for a specified number of epochs.
6.  **Model Evaluation:** Evaluating the trained model's performance (loss and accuracy) on the preprocessed test data (`x_test`, `y_test`).
7.  **Prediction & Visualization:**
    * Generating predictions for the test set.
    * Displaying several test images along with their actual labels and the model's predicted labels to qualitatively assess performance.

***

## Technologies Used

* **Python**
* **TensorFlow & Keras:** For building, compiling, training, and evaluating the Convolutional Neural Network.
* **NumPy:** For numerical operations, especially array reshaping and manipulation.
* **Matplotlib:** For visualizing sample images and results.
* **Jupyter Notebook / Google Colab:** For the interactive development environment.

***

## How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch.git](https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch.git)
    cd Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch
    ```
2.  **Install dependencies:**
    (It is recommended to use a virtual environment)
    ```bash
    pip install tensorflow numpy matplotlib jupyter
    ```
3.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook "8_Vision_AI_Fundamentals_Building_a_Digit_Recognizer_from_Scratch (2).ipynb"
    ```
    *(Run the cells sequentially. The notebook handles dataset download via the Keras API.)*

***

## Author & Portfolio Use

* **Author:** Jayasurya227
* **Portfolio:** This project ([https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch](https://github.com/Jayasurya227/Vision-AI-Fundamentals-Building-a-Digit-Recognizer-from-Scratch)) provides a clear demonstration of fundamental deep learning and computer vision skills applied to a classic problem. It is suitable for showcasing on GitHub, resumes/CVs, LinkedIn, and during interviews for roles involving AI, machine learning, or deep learning.
* **Notes:** Recruiters can review the code to understand the process of data preparation for image models, CNN architecture design using Keras layers, model training procedure, and performance evaluation for a classification task.
