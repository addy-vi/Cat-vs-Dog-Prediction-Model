# Dog and Cat Classification Using MobileNetV2

## Introduction
This project implements a **Dog and Cat classification** model using the **MobileNetV2** architecture, trained on the Kaggle "Dogs vs. Cats" dataset. MobileNetV2 is a lightweight deep learning model designed for efficient image classification, making it ideal for use in environments with limited resources. In this project, we use MobileNetV2 as a feature extractor, fine-tuning it on a dataset of dog and cat images to classify whether an image contains a dog or a cat.

## Objectives
- Train a classification model to differentiate between images of dogs and cats.
- Use **MobileNetV2**, a pre-trained model, for feature extraction and fine-tune it on the "Dogs vs. Cats" dataset.
- Achieve high classification accuracy by leveraging transfer learning.
- Provide insights into the model's performance and visualize the results.

## Technologies Used
- **Programming Language:** Python
- **Libraries/Frameworks:** TensorFlow, Keras, NumPy, Matplotlib, OpenCV
- **Tools:** Jupyter Notebook, VS Code
- **Dataset:** Kaggle's "Dogs vs. Cats" dataset (https://www.kaggle.com/c/dogs-vs-cats)

## Dataset Information
The **Dogs vs. Cats** dataset contains:
- **Training data**: 25,000 images of dogs and cats.
- **Test data**: 12,500 images of dogs and cats.


## Workflow

### 1. **Data Collection:**
   - Download the dataset from Kaggle's "Dogs vs. Cats" competition page.
   - The dataset is split into two parts: a training set and a test set. Each image is either labeled as "dog" or "cat."

### 2. **Data Preprocessing:**
   - **Resizing:** Resize images to a fixed size (e.g., 224x224 pixels) for MobileNetV2 input.
   - **Normalization:** Normalize image pixel values to a range of [0, 1].
   - **Data Augmentation:** Apply transformations like rotation, flipping, and zooming to prevent overfitting and increase model generalization.

### 3. **Model Development:**
   - **MobileNetV2**: Load the pre-trained MobileNetV2 model (trained on ImageNet) and exclude the top classification layer.
   - **Fine-tuning**: Add a new classifier on top of the MobileNetV2 base to adapt it for the dog and cat classification task.
   - **Model Compilation**: Compile the model using binary cross-entropy loss and the Adam optimizer.

### 4. **Training the Model:**
   - Train the model on the training dataset.
   - Use **early stopping** to prevent overfitting and **validation data** to monitor the model's performance during training.

### 5. **Model Evaluation:**
   - Evaluate the model on the test dataset to check its accuracy and visualize the predictions.
   - Plot accuracy and loss curves to monitor training progress.

## Results
- **Accuracy achieved on test dataset**: The model is expected to achieve over 97% accuracy after training.
- **Loss and accuracy curves**: Visualize the training and validation loss/accuracy to understand the model’s performance during training.

## Future Work
- **Fine-tuning**: Unfreeze additional layers in MobileNetV2 to fine-tune more parameters for better accuracy.
- **Model Optimization**: Convert the trained model to TensorFlow Lite or ONNX for deployment on mobile or edge devices.
- **Extend Dataset**: Add more categories or use a more complex dataset to create a multi-class image classification model.

## Conclusion
This project demonstrates how to classify images of dogs and cats using **MobileNetV2**, leveraging transfer learning to fine-tune a pre-trained model. The model achieves high classification accuracy and can be further extended for other image classification tasks.

## Clone Repository

To clone this repository to your local machine, run the following command:

```bash
git clone https://github.com/your-username/dog-cat-classification-mobilenetv2.git
