# ML Framework Lab: CIFAR-10 Image Classification Pipeline

This project demonstrates a complete Deep Learning workflow, emphasizing an isolated and reproducible environment using `uv` and modular PyTorch architecture.

## 🎯 Purpose
The goal of this lab is to demonstrate how modern tools can be used to create a consistent, reproducible environment and a modular pipeline for machine learning development.

## 🛠️ Tools and Technologies

- **Python** (Managed via `uv`)
- **PyTorch** (Core Deep Learning framework)
- **Pytorch Lightning** (High-level interface for modular PyTorch development)
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
│   └── cifar-10-batches-py.dvc
├── ML_lab/
│   └── check_env.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py  #deleted after using Pytorch Lightninhg
├── EDA.ipynb
├── .dvcignore
├── .gitignore
├── .python-version
├── app.py
├── export.py
├── verify_onnx.py
├── model.onnx.dvc
├── main.py
├── pyproject.toml
├── README.md
└── uv.lock          

```

🏗️ Project Architecture Update
To improve modularity and adhere to the principle of Boilerplate Reduction, I refactored the pipeline from vanilla PyTorch to PyTorch Lightning. In this professional architecture, the manual loops and evaluation logic previously found in `train.py` were integrated directly into the `LightningModule` methods (`training_step`, `validation_step`, `test_step`) within src/model.py. This makes the codebase significantly easier to maintain and scale.

## Performance Impact:
 Transitioning to the Lightning framework resulted in a test accuracy of 66.54% for the primary experiment (exp1), a notable improvement over the initial vanilla implementation.


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

I optimized the pipeline with BatchNorm and improved normalization, leading to a new performance baseline:

| Experiment | Name | Learning Rate | Batch Size | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| 01 | exp1 | 0.001 | 64 | 73.85% |
| 02 | exp2 | 0.01 | 64 | 64.48% |
| 03 | exp3 | 0.001 | 128 | 73.79% |

### ⚠️Note: 

Final accuracy results may vary slightly across different runs due to stochastic weight initialization and data shuffling, which is typical for Deep Learning pipelines.

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
### ⚠️ Note on Data Access
The raw data is managed via DVC and stored in a private AWS S3 bucket. If you do not have access to the remote storage, you can download the data directly via PyTorch by following these steps:

1. Open `src/dataset.py`.
2. Locate the `CIFAR10` dataset initialization.
3. Temporarily change `download=False` to `download=True`.
4. Run `main.py` once to download the dataset to the `data/` folder.
5. Change it back to `download=False` to maintain the pipeline integrity.

I have used this same method to initially fetch and then version-control the data with DVC.

## 🚀 Task 2: Model Deployment (In Progress)
I have successfully exported the best-performing model to the ONNX format for cross-platform inference.

* **Format**: ONNX (Open Neural Network Exchange).

* **Verification**: Passed via verify_onnx.py (Output matches PyTorch results).

* **Storage**: Model weights are tracked via DVC to keep the Git history clean.

To use the model, run:

```bash
dvc pull model.onnx.dvc
uv run python verify_onnx.py
```

## ✍️ Reflection

During this project, I learned how important it is to have a correctly configured and reproducible development environment. A major challenge was installing PyTorch with GPU support, as my RTX 50-series GPU required a test build with CUDA 12.8. This helped me understand how hardware and software compatibility affects machine learning workflows.

The verification script was kept intentionally simple to ensure clear failures if the environment is incorrect, making problems easier to detect. Solving the hardware alignment issues and modularizing the code into src/ provided deep insight into professional dependency management and troubleshooting in ML environments.