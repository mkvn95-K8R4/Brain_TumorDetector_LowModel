# 🧠 Brain Tumor Detection System
Please note this project was made for a college level MiniProject

An AI-powered web application that detects and classifies brain tumors from MRI images using deep learning techniques.
The system identifies tumor types such as **Glioma, Meningioma, Pituitary**, and also detects cases with **No Tumor**.

---

## Features

* Upload Brain MRI images through a web interface
* Classifies tumor into multiple categories
* Highlights suspected tumor region in the image
* Displays basic medical insights:
  * Description
  * Causes
  * Symptoms
  * Suggestions
* Simple and responsive UI
* Built using Deep Learning (CNN)

---

##  Tech Stack

* Python
* Flask (Web Framework)
* TensorFlow / Keras
* OpenCV
* NumPy

---

## 📂 Project Structure

```
brain_tumor_ai_project/
│
├── app.py                  # Main Flask application
├── train_model.py          # Model training script
├── brain_tumor_model.h5    # Trained model (NOT included in repo)
│
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Result display page
│
├── static/                 # Stores uploaded & processed images
│
├── dataset/                # Dataset folder (NOT included)
│   ├── Glioma/
│   ├── Meningioma/
│   ├── Pituitary/
│   └── No Tumor/
│
└── requirements.txt
```

---

##  Dataset Download (from Kaggle)

This project uses a publicly available Brain MRI dataset.

👉 Download from:
**Kaggle Brain Tumor MRI Dataset**

Steps:

1. Go to Kaggle and search for:
   `Brain Tumor MRI Dataset`
2. Download the dataset as ZIP
3. Extract it
4. Rename the folder to:

```
dataset
```

5. Place it inside your project directory

Make sure the structure looks like:

```
dataset/
 ├── Glioma/
 ├── Meningioma/
 ├── Pituitary/
 └── No Tumor/
```

---

##  Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/your-username/brain-tumor-detector.git
cd brain-tumor-detector
```

---

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## 🧠 Train the Model (Optional)

If you want to retrain the model:

```
python train_model.py
```

This will generate:

```
brain_tumor_model.h5
```

---

##  Run the Application

```
python app.py
```

Then open your browser:

```
http://127.0.0.1:5000
```

---

##  How It Works
1. User uploads MRI image
2. Image is preprocessed (resize + normalization)
3. CNN model predicts tumor type
4. OpenCV highlights suspected tumor region
5. Result is displayed with additional information

---

## Important Notes
* The dataset is **not included** due to size limitations
* The trained model (`.h5`) is also **not included**
* You can either:
  * Train your own model
  * Or use a pre-trained model

---

## Disclaimer
This project is for **educational purposes only**.
The predictions made by the model are not a substitute for professional medical diagnosis.
Always consult a qualified healthcare professional.


