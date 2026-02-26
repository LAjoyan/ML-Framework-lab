import torch
from src.model import SimpleCnn

def export():
    # Load the model from your best checkpoint
    model = SimpleCnn.load_from_checkpoint("path/to/your/best_model.ckpt")
    model.eval()

    # Define the input shape (Batch, Channels, Height, Width)
    input_sample = torch.randn((1, 3, 32, 32))

    # Export to ONNX
    model.to_onnx("model.onnx", input_sample, export_params=True)
    print("✅ Successfully exported model to ONNX format.")

if __name__ == "__main__":
    export()