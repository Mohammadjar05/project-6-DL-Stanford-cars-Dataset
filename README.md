# project-6-DL-Stanford-cars-Dataset
# 🚗 Car Image Classification Using Deep Learning

## 📌 Project Overview

This project uses **Deep Learning and Computer Vision** to identify a car model from an image.

The main idea is simple: the user provides an image of a car, and the trained model predicts which car model it is. After identifying the car, the system also displays additional information such as the manufacturer, country of origin, year, body type, horsepower, engine, top speed, transmission, and average price.

The project combines **car images with structured car information**, making the final result more useful than simply predicting a class name.

---

## 🎯 Project Goal

The goal of this project is to build an image classification system capable of recognizing different car models.

The system is designed to:

* Identify the car model from an image.
* Predict the top 5 possible car models.
* Display the model's confidence score.
* Retrieve additional information about the predicted car.
* Allow users to upload their own car image and receive a prediction.

The model recognizes **196 different car classes**.

---

## 📊 Dataset

The project uses the **Stanford Cars dataset** for the image classification part. The dataset contains images of different car models, together with annotations that specify the image filename, bounding box, and class ID.

In addition to the images, a CSV file called:

`stanford_cars_full_dataset.csv`

is used to store additional information about each car, including:

* Car name
* Manufacturer
* Model
* Year
* Country of origin
* Body type
* Horsepower
* Engine
* 0–60 mph time
* Top speed
* Transmission
* Average price

This allows the project to connect the image prediction with useful car specifications.

---

## 🛠️ Technologies and Libraries

The project was developed using Python and several machine learning and data science libraries:

* **PyTorch** – building and training the deep learning model.
* **Torchvision** – image transformations and the ResNet-50 architecture.
* **Pandas** – reading and processing the car information dataset.
* **NumPy** – numerical operations.
* **Matplotlib** – creating visualizations.
* **Seaborn** – data visualization.
* **Pillow (PIL)** – loading and processing images.
* **Scikit-learn** – splitting the dataset into training and validation sets.
* **SciPy** – reading the original MATLAB annotation files.
* **tqdm** – displaying training progress.
* **Kaggle** – downloading the dataset.

---

# 🔍 Project Workflow

The project can be divided into several main stages:

```text
Dataset Download
       ↓
Data Exploration
       ↓
Dataset Preparation
       ↓
Train / Validation / Test Split
       ↓
Image Preprocessing & Augmentation
       ↓
ResNet-50 Model
       ↓
Model Training
       ↓
Fine-Tuning
       ↓
Model Evaluation
       ↓
Car Image Prediction
       ↓
Display Car Information
```

---

## 1. 📥 Importing Libraries and Selecting the Device

The project starts by importing the required Python libraries.

PyTorch is also used to check whether a GPU is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If a CUDA-compatible GPU is available, the model runs on the GPU. Otherwise, it uses the CPU.

This is especially useful because training an image classification model such as ResNet-50 can be computationally expensive.

---

## 2. 📦 Downloading the Dataset

The Stanford Cars dataset is downloaded using Kaggle:

```python
!kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars
```

The downloaded ZIP file is then extracted into a directory containing the training and testing images.

The project also loads the CSV file containing the additional car information.

---

## 3. 📋 Exploring the Car Dataset

Pandas is used to load the CSV file:

```python
df = pd.read_csv("stanford_cars_full_dataset.csv")
```

Before training the model, the dataset is inspected to understand its structure.

The project checks for missing values and calculates some basic statistics, such as:

* Number of cars
* Number of countries
* Number of body types
* Year range
* Average horsepower
* Average price
* Most expensive car
* Cheapest car
* Most powerful car

This gives a better understanding of the dataset before starting the deep learning process.

---

# 📊 4. Data Visualization

Several visualizations are created to explore the car dataset.

### Cars by Country

The project counts how many car models come from each country and displays the results using a horizontal bar chart.

This helps show how the dataset is distributed across different countries.

### Average Price

The project also compares the average car price by:

* Country of origin
* Body type

This makes it easier to see differences in pricing between different categories of cars.

### Most Expensive and Most Powerful Cars

Another visualization shows:

* The top 10 most expensive cars.
* The top 10 cars with the highest horsepower.

These graphs provide a quick overview of the characteristics of the cars included in the dataset.

---

# 🗃️ 5. Creating a Car Database

A Python dictionary called `car_database` is created from the CSV data.

The purpose of this dictionary is to quickly retrieve information about a car after the neural network predicts its class.

For example, once the model predicts a specific car class, the program can use the predicted class ID to retrieve:

```text
Manufacturer
Model
Year
Country
Body Type
Horsepower
Engine
0–60 mph
Top Speed
Transmission
Average Price
```

This is what connects the **image classification model** to the **car information database**.

---

# 🏷️ 6. Processing the Stanford Cars Annotations

The Stanford Cars dataset provides annotation files in MATLAB `.mat` format.

The project uses `scipy.io.loadmat()` to read these files.

The annotations contain information such as:

* Image filename
* Bounding box coordinates
* Class ID

The annotation information is converted into Pandas DataFrames and saved as:

```text
anno_train.csv
anno_test.csv
```

This makes the annotation data easier to work with using Pandas.

---

# 🔢 7. Preparing Class Labels

The original class IDs range from **1 to 196**.

However, PyTorch classification labels normally start from zero.

Therefore, the project converts the labels:

```python
anno_train["label"] = anno_train["class_id"] - 1
```

This changes:

```text
1 → 0
2 → 1
3 → 2
...
196 → 195
```

As a result, the model works with **196 classes numbered from 0 to 195**.

---

# ✂️ 8. Splitting the Dataset

The training images are divided into:

* **Training set**
* **Validation set**

The test set remains separate.

The project uses:

```python
train_test_split(
    train_paths,
    train_labels,
    test_size=0.15,
    random_state=42,
    stratify=train_labels
)
```

The validation set contains **15% of the original training data**.

`stratify=train_labels` helps maintain a similar class distribution between the training and validation sets.

---

# 🖼️ 9. Image Preprocessing and Data Augmentation

Before images are given to the neural network, they need to be converted into a consistent format.

The images are resized and cropped to:

```text
224 × 224 pixels
```

For training images, several augmentation techniques are used:

* Random cropping
* Random horizontal flipping
* Random rotation
* Color changes
* Random grayscale
* Normalization

These transformations create slightly different versions of the training images.

This helps the model become less dependent on a specific image orientation, lighting condition, or appearance.

Validation and test images use a simpler transformation because we want to evaluate the model on images without random modifications.

---

# 🧩 10. Custom PyTorch Dataset

A custom `CarDataset` class is created using PyTorch's `Dataset`.

Its job is to:

1. Find an image.
2. Open the image.
3. Convert it to RGB.
4. Apply the required transformations.
5. Return the image and its label.

The project then uses `DataLoader` to load images in batches.

The batch size is:

```python
BATCH_SIZE = 32
```

Therefore, the model normally processes **32 images at a time**.

---

# 🧠 11. Deep Learning Model – ResNet-50

The main model used in this project is **ResNet-50**.

Instead of training ResNet-50 completely from the beginning, the project starts with a model that has already been trained on a large image dataset.

This is called **Transfer Learning**.

The original final classification layer is replaced with a new layer designed for the 196 car classes.

The final part of the network is:

```python
nn.Dropout(0.4)
nn.Linear(backbone.fc.in_features, num_classes)
```

The dropout layer helps reduce overfitting, while the linear layer produces predictions for all **196 car classes**.

---

# 🏋️ 12. Training Strategy

The model is trained in multiple phases rather than immediately training the entire network.

### Phase 1 – Train the Classification Head

During the first phase, most of the ResNet-50 layers are frozen.

Only the final classification layer is trained for **5 epochs**.

The idea is to first allow the new classification layer to learn how to distinguish between the 196 car classes.

---

### Phase 2 – Fine-Tuning

After the classification head has learned useful information, all layers of the model are unfrozen.

The entire network is then fine-tuned for **25 epochs**.

Different learning rates are used:

* Smaller learning rate for the pretrained ResNet layers.
* Larger learning rate for the new classification layer.

This allows the pretrained features to be adjusted without changing them too aggressively.

---

### Phase 3 – Low Learning Rate Fine-Tuning

The model is trained for another **10 epochs** using even smaller learning rates.

This final stage allows the model to make smaller adjustments to its learned features.

Overall, the training process can run for up to **40 epochs**.

---

# 📉 13. Loss Function

The project uses:

```python
nn.CrossEntropyLoss(label_smoothing=0.1)
```

Cross-Entropy Loss is commonly used for multi-class classification.

Since the model has 196 possible classes, the loss measures how close the model's predictions are to the correct class.

The `label_smoothing` parameter helps prevent the model from becoming too confident in its predictions and can help with generalization.

---

# ⚙️ 14. Optimizer and Learning Rate Scheduler

The project uses the **Adam optimizer** to update the model's weights.

During the fine-tuning stages, a:

```python
CosineAnnealingLR
```

learning rate scheduler is used.

The scheduler gradually changes the learning rate during training.

This allows the model to make larger updates earlier in training and smaller adjustments later.

---

# 📈 15. Model Evaluation

During training, the project records:

* Training loss
* Validation loss
* Training accuracy
* Validation accuracy

These values are stored in the `history` dictionary.

At the end, the training and validation curves are plotted.

The graphs help us understand how the model learned over time and whether the training and validation performance are moving in a similar direction.

---

# 💾 16. Saving the Best Model

The project keeps track of the best validation accuracy.

Whenever the model achieves a better validation result, its weights are saved:

```python
torch.save(model.state_dict(), "best_model.pth")
```

After training is finished, the best saved model is loaded before evaluating the test dataset.

This means the final evaluation uses the model that achieved the best validation performance rather than simply using the weights from the final training epoch.

---

# 🧪 17. Testing the Model

After loading the best model, the test dataset is evaluated.

The project reports:

```text
Best Validation Accuracy
Final Test Accuracy
```

The test accuracy represents the percentage of test images that the model classified correctly.

---

# 🚗 18. Making Predictions

The `predict()` function is responsible for recognizing a new car image.

The process is:

```text
Input Image
     ↓
Resize & Normalize
     ↓
ResNet-50
     ↓
Class Probabilities
     ↓
Predicted Car Model
     ↓
Car Database
     ↓
Car Information Report
```

The model uses Softmax to convert its output into probabilities.

The class with the highest probability becomes the predicted car model.

The function also retrieves the **top 5 predictions**, which gives the user several possible matches instead of only one result.

---

# 📋 19. Car Report

After prediction, the program displays a car report containing information such as:

```text
Car
Confidence
Manufacturer
Origin
Year
Body Type
Horsepower
Engine
0–60 mph
Top Speed
Transmission
Average Price
```

For example, the system might produce a result similar to:

```text
CAR REPORT
--------------------------------
Car          : Example Car
Confidence   : 92.4%
Manufacturer : Example Manufacturer
Origin       : Germany
Year         : 2020
Body Type    : Sedan
Horsepower   : 300 HP
Engine       : 3.0L
0-60 mph     : 5.2 sec
Top Speed    : 155 mph
Transmission : Automatic
Avg Price    : $65,000
```

The actual values depend on the image being analyzed and the model's prediction.

---

# 📤 20. Uploading a New Car Image

The final part of the project allows the user to upload an image directly in Google Colab.

The program asks the user to:

```text
Upload a car image...
```

After the image is uploaded, it is passed to the prediction function.

The system then:

1. Analyzes the image.
2. Predicts the car model.
3. Displays the confidence score.
4. Shows the car information.
5. Displays the uploaded image.

This makes the project interactive and allows users to test the trained model with their own car images.

---

# ⭐ Key Features

* 🚗 **196 car classes**
* 🧠 **ResNet-50 deep learning model**
* 🔄 **Transfer learning**
* 🖼️ **Image augmentation**
* 📊 **Data visualization**
* 📋 **Car specifications database**
* 🎯 **Top-5 predictions**
* 📈 **Training and validation accuracy tracking**
* 💾 **Best model checkpointing**
* 📤 **User image upload**
* ⚡ **GPU support with CUDA**

---

# 🧰 Project Structure

A simplified view of the important files generated or used by the project is:

```text
Project/
│
├── stanford_cars_full_dataset.csv
├── anno_train.csv
├── anno_test.csv
├── best_model.pth
│
├── extra_data/
│   └── car_data/
│       └── stanford_cars/
│           ├── cars_train/
│           └── cars_test/
│
└── DL_Project_CARsales.ipynb
```

---

# 🚀 Conclusion

This project demonstrates how **Computer Vision, Transfer Learning, and structured data** can be combined to create a practical car recognition system.

Instead of only predicting an image class, the system connects the prediction to a car database and provides additional information about the identified vehicle.

The project also demonstrates an end-to-end deep learning workflow, starting from dataset preparation and exploration, continuing through image preprocessing and model training, and ending with real-world image prediction.
