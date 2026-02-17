# ML Framework Lab

This project demonstrates how to set up an isolated and reproducible machine learning development environment using `uv`.

## Purpose

The purpose of this lab is to show how modern tools can be used to create a consistent and reproducible environment for machine learning development.
---

## Tools and Technologies

This project uses the following tools:

- Python
- uv 
- PyTorch
- Scikit-learn
- Pandas
- Jupyter Notebook
---
## Project Structure

```text
ML-Framework-Lab/
├─ ML_lab/
│  └─ check_env.py          
├─ .venv/               
├─ .gitignore
├─ .python-version     
├─ main.py             
├─ pyproject.toml       
├─ uv.lock              
└─ README.md            

```
---
## Quickstart (Run on any computer)

1) Install `uv` (once on your machine).
2) Clone this repository.
3) From the project root, run:

```bash
uv sync
uv run python ML_lab/check_env.py
```
---

## GPU / CUDA 

The environment supports GPU acceleration with PyTorch when CUDA is available.

On my computer (RTX 50-series), the standard stable PyTorch builds did not support the GPU architecture (sm_120).  
To make CUDA work, I had to install a PyTorch build from the test CUDA 12.8 wheel index.

This is configured through the `tool.uv.index` section in `pyproject.toml`:

- `https://download.pytorch.org/whl/test/cu128`

If the computer does not have an NVIDIA GPU or CUDA drivers, the project still runs on CPU.  
The script `ML_lab/check_env.py` prints whether CUDA is available.

If CUDA is not available, the script will automatically fall back to CPU.

## ✍️ Reflection

During this project, I learned how important it is to have a correctly configured and reproducible development environment.

A major challenge was installing PyTorch with GPU support, because my RTX 50-series GPU required a test build with CUDA 12.8. The standard installation did not work. This helped me understand how hardware and software compatibility affects machine learning workflows.

The verification script is intentionally simple. It focuses on:

- Checking the Python version

- Verifying installed package versions

- Detecting available hardware (CUDA or CPU)

- Running a small tensor calculation

I did not use advanced error handling (try/except), because if the environment is incorrectly set up, the script should fail clearly. This makes problems easier to detect.

If CUDA is not available, the script automatically falls back to CPU. This allows the project to run on different computers without modification.

For larger projects, possible improvements would be:

- Adding more detailed error messages

- Separating checks into functions

- Verifying CUDA driver versions

- Using CI to automatically test the environment

This project helped me understand dependency management, reproducibility, and troubleshooting of machine learning environments.

## Data Version Control (DVC)

This project uses DVC to track and version the dataset.

The raw data is stored in an AWS S3 bucket and is not included in Git.

### Setup

After cloning the repository, download the dataset with:

```bash
dvc pull
```

This will fetch the data from the configured S3 remote.

### Notes

* Raw data is stored in data/raw/

* Dataset files are tracked using DVC

* Large files are not committed to Git