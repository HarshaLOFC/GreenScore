# 🌱 GreenScore  
### Sustainability-Aware Benchmarking of Machine Learning Models

GreenScore is a Green AI evaluation system that benchmarks machine learning classification models not only on predictive performance, but also on **energy consumption** and **carbon emissions**.  
The project enables sustainability-aware model selection through an interactive Streamlit dashboard.

---

## 🚀 Live Demo
🔗 **(Add your Streamlit Cloud link here after deployment)**  
Example: https://greenscore.streamlit.app

---

## 🎯 Problem Statement
Traditional machine learning evaluation focuses mainly on accuracy while ignoring the environmental impact of training models.  
This project addresses the need for a framework that evaluates ML models using both **performance metrics** and **sustainability metrics** to support environmentally responsible AI development.

---

## 🧠 Key Features
- 🌿 GreenScore metric combining accuracy, energy, carbon emissions, and training time
- 🎚️ Priority-based weighting system (user-controlled)
- 📊 Visual comparison of models (tables & plots)
- 📂 Custom dataset upload (CSV)
- 🧪 Built-in controlled dataset for benchmarking
- ⚡ Energy & CO₂ tracking using CodeCarbon
- 🎨 Clean green–blue themed UI using Streamlit

---

## 🧩 System Architecture

```
Streamlit UI (app.py)
        ↓
User Inputs & Priorities
        ↓
Evaluation Pipeline
        ↓
Model Training + CodeCarbon
        ↓
GreenScore Computation
        ↓
Results Visualization
```

---

## 📁 Project Structure

```
GreenScore/
│
├── app.py                  # Streamlit entry point
├── requirements.txt        # Dependencies
├── dashboard/              # UI & plots
│   ├── ui_components.py
│   └── plots.py
│
├── pipeline/               # ML pipeline
│   ├── run_pipeline.py
│   └── preprocess.py
│
├── models/                 # ML models
│   ├── logistic.py
│   ├── random_forest.py
│   └── mlp.py
│
├── data/
│   ├── controlled/         # Built-in dataset
│   └── custom_dataset.py  # Custom dataset loader
│
├── utils/
│   └── metrics.py          # GreenScore logic
│
└── evaluation/             # Generated at runtime (ignored in Git)
```

---

## ⚙️ How GreenScore Works

Each model is evaluated using:
- **Accuracy**
- **F1 Score**
- **Energy Consumption (kWh)**
- **Carbon Emissions (tons CO₂)**
- **Training Time (seconds)**

All metrics are normalized and combined using user-defined priorities:

```
GreenScore =
  w_accuracy × Accuracy
+ w_energy   × (1 − Energy)
+ w_carbon   × (1 − CO₂)
+ w_time     × (1 − Time)
```

This allows different sustainability perspectives (eco-first vs accuracy-first).

---

## 🧪 Supported Models
- Logistic Regression
- Random Forest
- Neural Network (MLP)

---

## 📊 Dataset Options

### 1️⃣ Controlled Mode
- Built-in clean classification dataset
- Ensures fair and reproducible benchmarking

### 2️⃣ Custom Dataset Mode
- Upload your own CSV file
- Select target column
- Automatically validated for classification

**Requirements for custom datasets:**
- Numeric features only
- No missing values
- Classification target (≥ 2 classes)

---

## 🛠️ Tech Stack
- **Python**
- **Streamlit** (UI & deployment)
- **Scikit-learn** (ML models)
- **Pandas / NumPy** (data handling)
- **Plotly** (visualization)
- **CodeCarbon** (energy & carbon tracking)

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌍 Deployment
The application is designed to be deployed on **Streamlit Community Cloud**, providing a public, shareable link for demonstrations and portfolio use.

---

## 📌 Future Enhancements
- Regression model support
- Preset sustainability modes (Eco / Balanced / Accuracy)
- Model explainability (SHAP)
- Run history comparison
- PDF / CSV report export

---

## 👨‍💻 Team
**Team Lead:** Hash  
**Project:** GreenScore – Green AI Internship  

---

## 📜 License
This project is for academic and educational purposes.
