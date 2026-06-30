# 🌍 Eco-Health Risk Predictor Dashboard

An end-to-end Machine Learning project that utilizes a **Random Forest Classifier** to evaluate and predict regional public health risks based on environmental and air quality parameters. This project features a fully operational development pipeline built in a Jupyter Notebook and deployed via an interactive **Streamlit** dashboard.

---

## 📊 Project Features
* **End-to-End ML Pipeline:** Data analysis, smart automated feature/target alignment, and model evaluation.
* **Smart Feature Selection:** Automated extraction of target labels (`HealthImpactClass`) and drop configurations for analytical features.
* **Interactive Inference Engine:** Dynamic frontend sliders mapping feature ranges dynamically to deliver real-time operational risk predictions.
* **Intuitive Visualizations:** Interactive real-time performance evaluation featuring a Confusion Matrix heatmap.

---

## 🧬 Dataset Description
The model is trained on the **Air Quality and Health Impact Dataset**, incorporating key environmental telemetry indicators:
* **AQI (Air Quality Index):** Overall atmospheric pollution status.
* **Particulate Matter:** PM2.5 and PM10 concentration vectors.
* **Gaseous Vectors:** $NO_2$, $SO_2$, and $O_3$ levels.
* **Climate Factors:** Temperature, Humidity, and Wind Speed metrics.
* **Clinical Targets:** Respiratory, Cardiovascular cases, and Hospital Admissions to determine the comprehensive **Health Impact Class** (0 = Low, 1 = Moderate, 2 = High).

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python
* **Analysis & Frameworks:** Pandas, NumPy, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment Interface:** Streamlit Engine

---

## 🚀 Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sunbalsharif52-cyber/Eco-Health-Risk-Predictor.git
cd Eco-Health-Risk-Predictor
