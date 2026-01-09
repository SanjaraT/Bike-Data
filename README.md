📌 Project Overview

Predicted hourly bike rental demand in Seoul using weather, temporal, and environmental features. Formulated as a regression problem, the project implements a full ML pipeline from preprocessing to deep learning.

📂 Dataset

Source: UCI Machine Learning Repository
Target: Rented Bike Count
Features: Weather, time, and environmental variables
Non-numeric/redundant columns removed (e.g., Date, Season)

⚙️ Preprocessing

Dropped non-numeric/redundant features
Converted all features to numeric
Checked for missing values
Train-validation-test split
Standardized using Z-score scaling (fitted on training set only)
Exploratory Data Analysis
Scatter plots and correlation heatmaps to identify relationships
Insights guided model selection and feature importance

🧠 Models & Performance
Linear Regression	~465	~0.47
Neural Network Linear	Worse than linear	—
Deep Neural Network (128→64→32→1)	~318	~0.75

📌 Key points: DNN captured nonlinear patterns, outperforming linear models with strong generalization.
