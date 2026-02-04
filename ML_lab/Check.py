import sys
import torch
import sklearn
import pandas as pd
import ipykernel


def main():

    print(f"""*****Python version***** 
    Python version: {sys.version.split()[0]}
          """)

    print(f"""*****Package versions*****
    torch: {torch.__version__}")
    scikit-learn: {sklearn.__version__}")
    pandas: {pd.__version__}")    
    ipykernel: {ipykernel.__version__} 

""")
    print("\n--- Hardware / Accelerator ---")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    device = "cuda" if cuda_available else "cpu"
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Using device: {device}\n")


if __name__ == "__main__":
    main()
