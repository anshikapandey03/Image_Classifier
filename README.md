# 🧠 ResNet Vision — AI Image Classifier

ResNet Vision is an AI-powered image classification web application built using **ResNet-18, PyTorch, FastAPI, HTML, CSS, and JavaScript**.

The application allows users to upload an image and receive the model's predicted class along with its confidence score and top-5 predictions.

## ✨ Features

* 🖼️ Upload JPG, JPEG, or PNG images
* 🤖 ResNet-18 based image classification
* 📊 Top-5 prediction probabilities
* 🎯 Confidence score for the top prediction
* ⚡ FastAPI backend
* 🎨 Responsive and modern web interface
* 🧠 CIFAR-10 dataset with 10 image classes
* 💻 Supports CPU-based inference

## 🛠️ Technologies Used

### Machine Learning

* Python
* PyTorch
* Torchvision
* Hugging Face Transformers
* ResNet-18
* CIFAR-10

### Backend

* FastAPI
* Uvicorn
* Python Multipart

### Frontend

* HTML5
* CSS3
* JavaScript

### Image Processing

* Pillow

## 📁 Project Structure

```text
ResNet-Vision/
│
├── app.py
├── train_resnet.py
├── index.html
├── style.css
├── script.js
├── requirements.txt
├── .gitignore
├── README.md
│
├── data/
│   └── CIFAR-10 dataset
│
└── resnet_cifar10/
    ├── config.json
    ├── model.safetensors
    └── preprocessor_config.json
```

> `data/` and `resnet_cifar10/` are excluded from GitHub using `.gitignore` because the dataset and trained model can be large.

## 📊 CIFAR-10 Classes

The model is trained to classify images into 10 categories:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd ResNet-Vision
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

Run:

```bash
python train_resnet.py
```

The training script uses the CIFAR-10 dataset and trains the ResNet-18 based classifier.

After training, the model is saved in:

```text
resnet_cifar10/
```

## 🚀 Run the Application

Start the FastAPI server using:

```bash
python -m uvicorn app:app --reload
```

The application will run locally at:

```text
http://127.0.0.1:8000
```

Open the address in your browser.

## 🔍 How It Works

```text
User Uploads Image
        ↓
      FastAPI
        ↓
Image Preprocessing
        ↓
    ResNet-18
        ↓
  Classification
        ↓
Top-5 Predictions
        ↓
Confidence Scores
        ↓
Displayed on Web UI
```

## 📈 Model Performance

The model was trained on the CIFAR-10 dataset.

**Test Accuracy:** Add your final accuracy here

For example:

```text
Test Accuracy: XX.XX%
```

## ⚠️ Important Note

CIFAR-10 contains small **32×32 images** and only supports 10 predefined classes.

Therefore, images downloaded from Google or other sources may not always be classified correctly, especially when they differ significantly from the original CIFAR-10 images.

For best results, use images that clearly represent one of the ten CIFAR-10 categories.

## 🔮 Future Improvements

* Improve model accuracy through additional fine-tuning
* Add image history
* Add model performance visualization
* Deploy the application online
* Add drag-and-drop image upload
* Support additional datasets and classes
* Add explainable AI features such as Grad-CAM

## 👩‍💻 Author

**Anshika Pandey**

B.Tech — Information Technology

Interested in **Data Science, Artificial Intelligence, and Machine Learning**.

## 📄 License

This project is created for educational and learning purposes.
