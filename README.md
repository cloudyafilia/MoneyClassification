# 💵 MONEY TALKS — Indonesian Rupiah Classification

**MONEY TALKS** is an image classification project that uses **Deep Learning and Transfer Learning** to recognize Indonesian Rupiah banknotes from images.

The system classifies banknotes into **7 different denominations**:

* Rp1.000
* Rp2.000
* Rp5.000
* Rp10.000
* Rp20.000
* Rp50.000
* Rp100.000

The project uses **MobileNetV2 pretrained on ImageNet** as the backbone model and fine-tunes it for Indonesian Rupiah banknote classification.

In addition to image classification, the project includes an **audio output feature** using Google Text-to-Speech (gTTS), allowing the predicted denomination to be converted into Indonesian speech.

The final model achieved a **test accuracy of 99.79%** on 2,328 test images.

---

## 🎯 Project Objectives

The main objectives of this project are:

1. Develop an image classification model for Indonesian Rupiah banknotes.
2. Classify banknotes into seven different denominations.
3. Apply transfer learning using MobileNetV2.
4. Improve model generalization through image augmentation.
5. Evaluate model performance using a test dataset.
6. Develop an inference pipeline for classifying newly uploaded images.
7. Convert the classification result into Indonesian speech using gTTS.

---

## 💡 Problem Statement

Recognizing banknote denominations can be challenging when relying solely on visual inspection, particularly for users who may have difficulty distinguishing between different denominations.

This project explores how **computer vision and deep learning** can be used to automatically identify Indonesian Rupiah banknotes from images.

The system takes an image of a banknote as input and produces:

```text
Image
  ↓
Image Preprocessing
  ↓
MobileNetV2
  ↓
Classification
  ↓
Predicted Denomination
  ↓
Text-to-Speech
  ↓
"Ini uang 20000"
```

The notebook demonstrates this workflow through an inference example in which an uploaded image is classified as `20000`, followed by the generated Indonesian speech **"Ini uang 20000"**.

---

# 📊 Dataset

The project uses a dataset of Indonesian Rupiah banknote images stored in folders according to their denominations.

The dataset contains **23,253 images** across 7 classes.

### Classes

| Class     | Denomination | Total Images |
| --------- | -----------: | -----------: |
| 1000      |      Rp1.000 |        3,262 |
| 2000      |      Rp2.000 |        3,257 |
| 5000      |      Rp5.000 |        3,358 |
| 10000     |     Rp10.000 |        3,339 |
| 20000     |     Rp20.000 |        3,332 |
| 50000     |     Rp50.000 |        3,347 |
| 100000    |    Rp100.000 |        3,358 |
| **Total** |              |   **23,253** |

The class distribution is relatively balanced, with each denomination containing approximately 3,200–3,400 images.

---

## 🖼️ Image Characteristics

The notebook checks image resolutions before training.

The dataset contains images with varying resolutions, including examples such as:

```text
960 × 1280
1600 × 708
720 × 1280
3024 × 4032
1600 × 900
1200 × 1600
1280 × 720
4160 × 3120
```

This variation motivates the use of a standardized input size during model training.

The model ultimately receives images resized to:

```text
224 × 224 × 3
```

which matches the expected input dimensions of MobileNetV2.

---

# 🔄 Data Splitting

The dataset is divided into three subsets using a fixed random seed of `42`:

* **80% Training**
* **10% Validation**
* **10% Testing**

The splitting process is performed independently for each denomination to maintain the class structure across the subsets.

The resulting split is:

| Class  | Train | Validation | Test |
| ------ | ----: | ---------: | ---: |
| 1000   | 2,609 |        326 |  327 |
| 2000   | 2,605 |        326 |  326 |
| 5000   | 2,686 |        336 |  336 |
| 10000  | 2,671 |        334 |  334 |
| 20000  | 2,665 |        333 |  334 |
| 50000  | 2,677 |        335 |  335 |
| 100000 | 2,686 |        336 |  336 |

---

# 🧹 Data Preparation

Several preprocessing steps are performed before training.

### 1. Image Resizing

Images are resized to:

```text
224 × 224 pixels
```

to match the MobileNetV2 input configuration.

### 2. RGB Conversion

Images are converted into RGB format during the inference process.

### 3. Batch Processing

The training dataset uses a batch size of **64**, while the validation dataset uses a batch size of **32**.

### 4. Class Weighting

Class weights are calculated using:

```python
compute_class_weight(
    class_weight='balanced'
)
```

and incorporated during model training.

---

# 🔀 Data Augmentation

To improve the model's ability to generalize to different image conditions, several augmentation techniques are implemented:

```text
RandomFlip
RandomRotation
RandomZoom
RandomContrast
RandomTranslation
RandomBrightness
Resizing
```

The specific configuration includes:

| Augmentation | Configuration         |
| ------------ | --------------------- |
| Random Flip  | Horizontal + Vertical |
| Rotation     | 0.1                   |
| Zoom         | 0.2                   |
| Contrast     | 0.2                   |
| Translation  | 0.1 × 0.1             |
| Brightness   | 0.2                   |
| Resize       | 224 × 224             |

These transformations expose the model to variations in orientation, scale, position, brightness, and contrast.

---

# 🧠 Model Architecture

The project uses **Transfer Learning with MobileNetV2**.

MobileNetV2 is initialized with weights pretrained on **ImageNet**:

```python
MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
```

The backbone is partially fine-tuned:

* MobileNetV2 is initially made trainable.
* The first **100 layers are frozen**.
* The remaining layers are trainable.

---

## 🏗️ Model Structure

The classification model consists of:

```text
Input Image
     │
     ▼
Rescaling
     │
     ▼
MobileNetV2
     │
     ▼
Batch Normalization
     │
     ▼
Global Average Pooling
     │
     ▼
Dense Layer
256 neurons + ReLU
     │
     ▼
Dropout
40%
     │
     ▼
Dense Output Layer
7 classes + Softmax
```

The implemented architecture is:

```python
model = models.Sequential([
    layers.Rescaling(scale=1./127.5, offset=-1),
    base_model,
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),
    layers.Dense(
        256,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.01)
    ),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])
```

---

## 📐 Model Parameters

The model contains:

| Parameter                |         Value |
| ------------------------ | ------------: |
| Total Parameters         |     2,592,839 |
| Trainable Parameters     |     2,193,735 |
| Non-trainable Parameters |       399,104 |
| Input Size               | 224 × 224 × 3 |
| Output Classes           |             7 |

---

# ⚙️ Model Compilation

The model is compiled using:

```python
optimizer = Adam(learning_rate=1e-4)
loss = categorical_crossentropy
metric = accuracy
```

Specifically:

| Configuration     | Value                    |
| ----------------- | ------------------------ |
| Optimizer         | Adam                     |
| Learning Rate     | 0.0001                   |
| Loss Function     | Categorical Crossentropy |
| Evaluation Metric | Accuracy                 |
| Output Activation | Softmax                  |

---

# ⏱️ Training Strategy

The model is configured for a maximum of **100 epochs**.

Three callbacks are used:

### Early Stopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

This stops training when validation loss does not improve for 10 consecutive epochs and restores the best weights.

### Reduce Learning Rate

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,
    min_lr=1e-6
)
```

The learning rate is reduced when validation loss stops improving.

### Model Checkpoint

```python
ModelCheckpoint(
    'best_model.weights.h5',
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=True
)
```

This stores the model weights associated with the best validation loss.

---

# 📈 Training Results

The training process shows rapid improvement during the first several epochs.

For example:

| Epoch | Train Accuracy | Validation Accuracy |
| ----: | -------------: | ------------------: |
|     1 |         76.25% |              77.84% |
|     2 |         98.66% |              95.00% |
|     3 |         99.79% |              97.80% |
|     4 |         99.93% |              98.55% |
|     5 |         99.94% |              98.66% |
|     6 |         99.98% |              98.95% |
|     7 |         99.91% |              99.09% |
|     8 |         99.88% |              99.49% |
|     9 |         99.98% |              99.38% |
|    10 |         99.92% |              99.41% |

The model continued training beyond these initial epochs, with validation accuracy reaching around **99.7%** in later epochs.

---

# 🏆 Model Evaluation

The final model is evaluated using a separate test dataset containing:

```text
2,328 images
7 classes
```

The evaluation includes:

* Confusion Matrix
* Classification Report
* Test Loss
* Test Accuracy

---

## 📊 Test Performance

The final evaluation produced:

| Metric            |     Result |
| ----------------- | ---------: |
| **Test Accuracy** | **99.79%** |
| **Test Loss**     | **0.0079** |
| Test Samples      |      2,328 |

This means that the trained MobileNetV2 model correctly classified approximately **99.79% of the test images**.

---

# 📋 Classification Report

The classification report shows extremely strong performance across all seven denominations.

| Denomination | Precision | Recall | F1-Score | Support |
| ------------ | --------: | -----: | -------: | ------: |
| Rp1.000      |      1.00 |   0.99 |     1.00 |     327 |
| Rp2.000      |      1.00 |   1.00 |     1.00 |     334 |
| Rp5.000      |      1.00 |   1.00 |     1.00 |     336 |
| Rp10.000     |      1.00 |   1.00 |     1.00 |     326 |
| Rp20.000     |      1.00 |   1.00 |     1.00 |     334 |
| Rp50.000     |      1.00 |   1.00 |     1.00 |     336 |
| Rp100.000    |      1.00 |   1.00 |     1.00 |     335 |

Overall:

```text
Accuracy     : 1.00
Macro F1     : 1.00
Weighted F1  : 1.00
```

The underlying `model.evaluate()` result gives the more precise test accuracy of **0.9979**.

---

# 🔍 Confusion Matrix

A confusion matrix is generated to analyze classification errors across the seven denominations.

```python
cm = confusion_matrix(
    y_true_labels,
    y_pred_labels
)
```

The resulting matrix is visualized using a Seaborn heatmap with:

* True labels on the y-axis.
* Predicted labels on the x-axis.

This provides a detailed view of which denominations are correctly classified and where misclassification occurs.

---

# 🔮 Inference

After training, the model can classify new images uploaded by the user.

The inference pipeline performs:

```text
Upload Image
     ↓
Convert to RGB
     ↓
Resize to 224 × 224
     ↓
Convert to Array
     ↓
Model Prediction
     ↓
Find Highest Probability Class
     ↓
Display Predicted Denomination
```

The notebook supports uploading multiple images at once.

---

# 🔊 Voice Output

One of the most interesting features of this project is the **text-to-speech output**.

After obtaining the predicted denomination, the system creates a sentence:

```python
prediction_text = f"Ini uang {predicted_class}"
```

The text is then converted into Indonesian speech using:

```python
gTTS(
    text=prediction_text,
    lang='id'
)
```

The resulting audio is played directly inside Google Colab.

For example:

```text
Input:
Image of Rp20.000 banknote

Prediction:
20000

Voice Output:
"Ini uang 20000"
```

The notebook demonstrates this exact inference result.

---

# 🌐 Model Export

The trained model is saved in HDF5 format:

```text
final_model_mobilenetv2.h5
```

The notebook also exports the model to the TensorFlow SavedModel format:

```text
saved_model
```

and converts the model into TensorFlow.js format:

```text
tfjs_model
```

This opens the possibility of deploying the classifier outside the original Google Colab environment.

---

# 🛠️ Technologies

The project uses:

### Programming

* Python

### Deep Learning

* TensorFlow
* Keras
* MobileNetV2

### Data Processing

* NumPy
* Pandas
* Pillow

### Visualization

* Matplotlib
* Seaborn

### Machine Learning Utilities

* Scikit-learn

### Deployment / Model Conversion

* TensorFlow.js

### Text-to-Speech

* gTTS

### Environment

* Google Colab
* GPU T4

## The notebook explicitly uses a T4 GPU configuration and imports the required TensorFlow/Keras, Scikit-learn, visualization, image processing, gTTS, and TensorFlow.js utilities.

# 📁 Repository Structure

The main notebook for this project is:

```text
MONEY_TALKS.ipynb
```

The notebook contains the complete workflow:

```text
MONEY_TALKS.ipynb
│
├── Import Libraries
├── Loading Data
├── Data Visualization
├── Data Splitting
├── Data Preparation
├── Data Augmentation
├── MobileNetV2 Model
├── Model Training
├── Model Evaluation
├── Confusion Matrix
├── Classification Report
├── Model Saving
├── Model Export
├── Image Inference
└── Text-to-Speech
```

---

# 🚀 How to Run

## 1. Open the Notebook

The project was developed in Google Colab.

Open:

```text
MONEY_TALKS.ipynb
```

and run the cells sequentially.

---

## 2. Install Dependencies

The notebook installs several required packages:

```bash
pip install gdown
pip install gTTS
pip install tensorflowjs
```

The remaining dependencies are provided by the TensorFlow/Google Colab environment.

---

## 3. Download Dataset

The notebook downloads the dataset from Google Drive using `gdown`.

The downloaded archive is:

```text
dataset.zip
```

and is extracted into:

```text
/content/rupiah_dataset/
```

The main dataset directory is:

```text
/content/rupiah_dataset/Data_Rupiah_Baru
```

The downloaded dataset archive is approximately **4.89 GB**, so sufficient storage is required to reproduce the notebook exactly.

---

## 4. Train the Model

Run the modeling section to:

```text
Load Dataset
     ↓
Calculate Class Weights
     ↓
Apply Augmentation
     ↓
Load MobileNetV2
     ↓
Fine-Tune Model
     ↓
Train
     ↓
Save Best Weights
     ↓
Save Final Model
```

---

## 5. Test the Model

The evaluation section generates:

* Confusion Matrix
* Classification Report
* Test Accuracy
* Test Loss

---

## 6. Try New Images

Run the inference section and upload one or more banknote images.

The model will return:

```text
Gambar: filename.jpg => Prediksi: 20000
```

and generate:

```text
Ini uang 20000
```

as Indonesian speech.

---

# 💼 Potential Applications

This project can be developed into several real-world applications, particularly for **assistive technology** and financial accessibility.

Potential applications include:

### 👁️ Assistive Banknote Recognition

A camera-based application could recognize the denomination of a banknote and announce it through audio.

### 📱 Mobile Application

The TensorFlow.js export could potentially be integrated into a web-based application, while the model itself could also be adapted for mobile deployment.

### 💳 Cash Transaction Assistance

The system could help users verify the denomination of a banknote during cash transactions.

### 🔊 Audio-Based Accessibility

The combination of image classification and text-to-speech makes the system suitable for applications where visual information needs to be communicated through audio.

---

# ⚠️ Limitations

Despite achieving very high test accuracy, several limitations should be considered.

### 1. Dataset Dependency

The model performance is dependent on the characteristics of the training dataset.

Real-world images may contain:

* Different lighting conditions.
* Background clutter.
* Partial occlusion.
* Folded or damaged banknotes.
* Unusual camera angles.
* Multiple banknotes in one image.

### 2. Dataset Size

The original dataset is large, but the test set contains 2,328 images and comes from the same overall dataset distribution. Therefore, additional external validation would be useful before claiming equivalent performance in real-world environments.

### 3. Inference Preprocessing

The notebook resizes uploaded images to 224 × 224 before prediction.

Future versions could make the inference preprocessing pipeline more explicitly consistent with the training preprocessing.

### 4. Deployment

The current implementation is demonstrated primarily in Google Colab. A production-ready version would require deployment into a dedicated web, desktop, or mobile application.

---

# 🔮 Future Improvements

Several improvements could make the project more practical.

### 📱 1. Mobile Deployment

Convert the trained model into a mobile-compatible format and integrate it with an Android application.

### 🎥 2. Real-Time Camera Detection

Instead of uploading individual images, the system could process frames from a smartphone camera in real time.

### 💵 3. Multiple Banknote Detection

Future versions could detect several banknotes simultaneously instead of classifying a single image.

### 🔊 4. Improved Voice Interaction

The voice output could be expanded to provide more useful information, for example:

```text
"Ini uang dua puluh ribu rupiah."
```

rather than simply reading the numerical class.

### 🌍 5. Real-World Validation

Evaluate the model using independently collected images from different environments, cameras, lighting conditions, and backgrounds.

### 🚀 6. Web/Mobile Deployment

The TensorFlow.js model export provides a starting point for browser-based deployment.

---

# 📌 Key Takeaways

This project demonstrates an end-to-end **Computer Vision and Deep Learning workflow** for Indonesian Rupiah banknote classification.

### Dataset

**23,253 images** across **7 denominations**.

### Model

**MobileNetV2 Transfer Learning** with partial fine-tuning.

### Input

**224 × 224 × 3 RGB images**.

### Evaluation

**2,328 test images**.

### Performance

**99.79% test accuracy** with a test loss of **0.0079**.

### Additional Feature

Predicted denominations can be converted into **Indonesian speech using gTTS**.

---

# 👩🏻‍💻 Author

**Cloudya Filia Putri**

Statistics Student | Data Analytics & Machine Learning Enthusiast

---

## 🔗 Project

[GitHub — MoneyClassification](https://github.com/cloudyafilia/MoneyClassification)

[Google Colab — MONEY_TALKS](https://colab.research.google.com/github/cloudyafilia/Klasifikasi-Gambar-Dicoding/blob/main/MONEY_TALKS.ipynb)

---

## 🏷️ Topics

`Python` `Computer Vision` `Image Classification` `Deep Learning` `Transfer Learning` `MobileNetV2` `TensorFlow` `Keras` `Indonesian Rupiah` `Banknote Classification` `gTTS` `TensorFlow.js` `Google Colab` `Machine Learning`
