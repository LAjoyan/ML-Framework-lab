import torch
from src.model import SimpleCnn

def export():
    # Initialize the model structure
    model = SimpleCnn(num_classes=10)
    
    # 2. Set to evaluation mode
    model.eval()

    # Create dummy input for CIFAR-10 shape (Batch, Channels, H, W)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Export to ONNX
    model.to_onnx("model.onnx", dummy_input, export_params=True)
    print("✅ Successfully exported model to model.onnx")

if __name__ == "__main__":
    export()