# Stock Price Forecasting using Generative Adversarial Networks

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-TensorFlow-orange.svg)

A deep learning project that leverages Generative Adversarial Networks (GANs) and Wasserstein GANs with Gradient Penalty (WGAN-GP) to forecast financial time-series data. This repository provides a complete pipeline from data preprocessing to model training and prediction.

---

## 📖 Table of Contents
- [✨ Project Overview](#-project-overview)
- [🎯 Key Features](#-key-features)
- [🛠️ Technology Stack](#️-technology-stack)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [⚙️ How to Use](#️-how-to-use)
- [📊 Results & Visualizations](#-results--visualizations)
- [👤 Author](#-author)
- [⚠️ Disclaimer](#️-disclaimer)

---

## ✨ Project Overview

Predicting stock market movements is a challenging task due to the data's inherent volatility and complex, non-linear patterns. This project implements advanced generative models to tackle this challenge. By comparing a baseline GRU model with a standard GAN and the more stable WGAN-GP, we aim to generate realistic future stock price scenarios that capture the nuances of market dynamics.

---

## 🎯 Key Features

- **Data Preprocessing**: Robust scripts to clean, normalize, and prepare time-series data for deep learning models.
- **Baseline Model**: A Gated Recurrent Unit (GRU) network is provided to establish a performance benchmark.
- **Advanced Generative Models**:
    - **GAN**: Implementation of a standard Generative Adversarial Network for time-series generation.
    - **WGAN-GP**: Implementation of a Wasserstein GAN with Gradient Penalty for improved training stability and prevention of mode collapse.
- **Forecasting & Evaluation**: Scripts to test trained models and generate multi-step future price predictions.
- **Automated Visualizations**: Automatically saves comparison plots of actual vs. predicted prices to showcase model performance visually.

---

## 🛠️ Technology Stack

- **Python 3.8+**
- **TensorFlow / Keras** for building and training neural networks.
- **NumPy** for numerical operations.
- **Pandas** for data manipulation and analysis.
- **Matplotlib** for data visualization.
- **Scikit-learn** for data preprocessing and evaluation metrics.

---

## 🚀 Getting Started

Follow these instructions to set up the project on your local machine.

### Prerequisites

- Python 3.8 or a newer version.
- `pip` package installer.

### Installation

1.  **Clone the repository to your local machine:**
    ```sh
    git clone https://github.com/barshal-horse/your-repository-name.git
    cd your-repository-name
    ```

2.  **Create and activate a virtual environment (highly recommended):**
    ```sh
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies from the `requirements.txt` file:**
    ```sh
    pip install -r requirements.txt
    ```

---

## ⚙️ How to Use

The project is structured into sequential Python scripts located in the `/Code` directory. Run them from your terminal in the following order.

1.  **Prepare the Data**
    This script will clean, normalize, and structure your raw stock data for training.
    ```sh
    python "Code/2. data_preprocessing.py"
    ```

2.  **Train the Baseline GRU Model (Optional)**
    Run this to train the GRU model and establish a performance benchmark.
    ```sh
    python "Code/3. Baseline_GRU.py"
    ```

3.  **Train a Generative Model**
    Choose between the standard GAN or the more advanced WGAN-GP for training.
    ```sh
    # To train the standard GAN
    python "Code/4. Basic_GAN.py"

    # To train the Wasserstein GAN with Gradient Penalty
    python "Code/5. WGAN_GP.py"
    ```

4.  **Test and Generate Forecasts**
    Use a trained model to generate and visualize future stock price predictions.
    ```sh
    python "Code/6. Test_prediction.py"
    ```

---

## 📊 Results & Visualizations

After running the testing script (`6. Test_prediction.py`), the model's performance will be evaluated.

- The key finding is that GAN-based models can produce highly realistic stock price forecasts, with **WGAN-GP** often demonstrating superior stability and predictive accuracy.
- Comparison plots that visualize the **Actual vs. Predicted** stock prices are automatically saved as `.png` image files inside the `/Code/` directory for your review.

---

## 👤 Author

**barshal-horse**
- **GitHub:** [https://github.com/barshal-horse](https://github.com/barshal-horse)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. The information and predictions generated by these models should **not** be considered financial advice. Trading in financial markets involves substantial risk, and you should not make investment decisions based solely on the output of this software.
