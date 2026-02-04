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
The script `ML_lab/check-env.py` prints whether CUDA is available.