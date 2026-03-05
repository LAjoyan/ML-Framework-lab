import torch
from src.model import SimpleCnn

def export():
    # Initialize the model structure
    model = SimpleCnn(num_classes=10)

    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    
    # 2. Set to evaluation mode
    model.eval()

    # Create dummy input for CIFAR-10 shape (Batch, Channels, H, W)
    dummy_input = torch.randn(1, 3, 32, 32)

    # Export to ONNX
    torch.onnx.export(
        model, 
        dummy_input, 
        "model.onnx", 
        export_params=True,
        input_names=['input'],
        output_names=['output']
    )
    print("✅ Successfully exported TRAINED model to model.onnx")
if __name__ == "__main__":
    export()