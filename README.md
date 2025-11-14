# 🩺 End-to-End Deep Learning Project – Skin Condition Multi-Label Classifier

This project builds a multi-label skin-condition classifier using a VGG16 pretrained model, frozen CNN layers, and a custom ANN classification head. The training pipeline is fully modular with YAML-based configs, automated data ingestion, model preparation, training, and prediction pipelines.The final model achieves strong accuracy for all 10 classes and is deployed through a Flask web app that performs real-time image prediction.Model achieved a training binary accuracy of 91.70% and a validation binary accuracy of 93.99% in 1 epoch only, demonstrating strong multi-label performance across all 10 skin-condition categories.



### 📦 Dataset  
Google Drive Download Link:  
https://drive.google.com/file/d/1XaNxpHP3XwDyKjEw-1wirLcgLqMRSsV-/view?usp=sharing

---

## ✅ Workflows Completed

1. Update `config.yaml`
2. Update `params.yaml`
3. Create entity classes (`config_entity.py` & `artifact_entity.py`)
4. Implement Configuration Manager (`configuration.py`)
5. Build Data Ingestion Component  
6. Build Base Model Preparation Component  
7. Build Model Training Component  
8. Build Training Pipeline  
9. Build Prediction Pipeline  
10. Build Flask App + HTML UI  
11. Run end-to-end system successfully

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

```bash
git clone <repo-link>
cd <project-folder>
``` 
### 2️⃣ Create & Activate Conda Environment
```bash
conda create -n cnn python=3.10 -y
conda activate cnn
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

4️⃣ Run the Full Training Pipeline

```bash
python main.py
```

5️⃣ Run the Flask App
```bash
python app.py
```

#### Then open your browser
```bash
http://127.0.0.1:5000/predict
```


## 🧠 Model Details

<img width="1917" height="1074" alt="model trained" src="https://github.com/user-attachments/assets/88dfc9e7-8972-4ea2-99f9-0c128ca40379" />

- **Model:** VGG16 pretrained on ImageNet  
- **include_top:** False  
- **Frozen Layers:** All convolutional layers  
- **Custom Head:**  
  - Dense → ReLU  
  - Dropout  
  - Dense → Sigmoid (multi-label output)  
- **Loss Function:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Learning rate:** from `params.yaml`  
- **Classes:** 10 skin conditions  


## 🖥️ Web App Preview
<img width="1913" height="1040" alt="ui prediction" src="https://github.com/user-attachments/assets/41a6e0f1-471b-466c-b8b3-72e6ed769866" />
A simple Flask
-based UI that supports:

- Uploading an image  
- Displaying the uploaded image  
- Showing predictions for each skin condition:  
  - Probability values  
  - "Yes/No" prediction  
  - Colour-coded table (Green = Yes, Red = No)  

Ideal for demonstrations, testing, or local deployment.


## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **Pandas / NumPy**
- **OpenCV / Pillow**
- **Flask**
- **YAML (configuration-driven pipeline)**
- **Modular MLOps-style folder structure**

