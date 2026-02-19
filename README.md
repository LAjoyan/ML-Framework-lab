# ML Framework Lab: CIFAR-10 Image Classification Pipeline

This project demonstrates a complete Deep Learning workflow, emphasizing an isolated and reproducible environment using `uv` and modular PyTorch architecture.

## 🎯 Purpose
The goal of this lab is to demonstrate how modern tools can be used to create a consistent, reproducible environment and a modular pipeline for machine learning development.

## 🛠️ Tools and Technologies

- **Python** (Managed via `uv`)
- **PyTorch** (Core Deep Learning framework)
- **DVC** (Data Version Control for AWS S3)
- **Scikit-learn & Pandas** (Data processing and environment verification)
- **Jupyter Notebook** (For EDA)

## 🏗️ Project Structure

The project follows a modular design to separate data handling, model architecture, and training logic:

```text
ML-Framework-Lab/
├── .dvc/
├── .venv/
├── data/
│   ├── cifar-10-batches-py/
│   ├── .gitignore
│   └── cifar-10-batches-py.dvc
├── ML_lab/
│   └── check_env.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── EDA.ipynb
├── .dvcignore
├── .gitignore
├── .python-version
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock          

```

## 🛠️ Task 0: Environment Verification & GPU Setup
Before building the pipeline, I developed a verification script to ensure hardware acceleration and dependency integrity. 

**Key Features:**

* **Hardware Detection:** Identifies CUDA availability for modern architectures.
* **Tensor Benchmark:** Performs a 1000 x 1000 matrix multiplication to verify compute functionality.
* **Version Auditing:** Reports exact versions for Python, Torch, Scikit-learn, and Pandas.

**GPU / CUDA Support (RTX 50-series)**

On my machine (RTX 50-series), standard PyTorch builds did not support the new architecture. I configured pyproject.toml to use the CUDA 12.8 test wheel index to enable GPU support:
https://download.pytorch.org/whl/test/cu128

To verify your environment:

```bash
uv sync
uv run python ML_lab/check_env.py
```
---

## 📊 Task 1: Experiments & Results

I conducted three experiments to evaluate the impact of Learning Rate and Batch Size on the CIFAR-10 dataset.

| Experiment | Name | Learning Rate | Batch Size | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 01 | exp1 | 0.001 | 64 | 64.14% |
| 02 | exp2 | 0.01 | 64 | 9.98% |
| 03 | exp3 | 0.001 | 128 | 63.51% |

### Analysis

* Convergence Failure: In `exp2`, a learning rate of 0.01 was too high for the Adam optimizer, preventing the model from converging and resulting in random-chance accuracy (~10%).

* Batch Size: Increasing the batch size to 128 (`exp3`) slightly reduced the learning progress compared to 64 within the same 3-epoch window.

* Reproducibility: All experiments are automated and can be triggered via:

```bash
uv run python main.py
```

## 📦 Data Version Control (DVC)
Raw data is stored in an AWS S3 bucket and tracked via `.dvc` files to keep the Git repository lightweight.

* Setup: To fetch the dataset after cloning, run: 

```bash
dvc pull
```

## ✍️ Reflection

During this project, I learned how important it is to have a correctly configured and reproducible development environment. A major challenge was installing PyTorch with GPU support, as my RTX 50-series GPU required a test build with CUDA 12.8. This helped me understand how hardware and software compatibility affects machine learning workflows.

The verification script was kept intentionally simple to ensure clear failures if the environment is incorrect, making problems easier to detect. Solving the hardware alignment issues and modularizing the code into src/ provided deep insight into professional dependency management and troubleshooting in ML environments.