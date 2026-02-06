import sys
import torch
import sklearn
import pandas as pd
import ipykernel


def main():

    python_version = sys.version.split()[0]
    torch_version = torch.__version__
    scikit_learn_version = sklearn.__version__
    pandas_version = pd.__version__
    ipykernel_version = ipykernel.__version__

    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"

    x = torch.randn(1000, 1000, device=device)
    y = x @ x
    if cuda_available:
        torch.cuda.synchronize()
    mean_result = y.mean().item()

    print(f"""\
==================================================
🔍 SYSTEM & ENVIRONMENT REPORT
==================================================
🐍 Python Version      : {python_version}
🔥 Torch               : {torch_version}
📈 Scikit-learn        : {scikit_learn_version}
🐼 Pandas              : {pandas_version}
🔌 IPyKernel           : {ipykernel_version}
--------------------------------------------------
🚀 CUDA Available      : {cuda_available}
💻 GPU                 : {gpu_name}
⚙️  Device              : {device}
--------------------------------------------------
📊 Tensor Mean Result  : {mean_result:.6f}
==================================================
""")


if __name__ == "__main__":
    main()
