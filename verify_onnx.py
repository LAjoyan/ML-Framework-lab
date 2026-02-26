import onnxruntime as ort
import numpy as np

try:
    # Use the session to load your model
    session = ort.InferenceSession("model.onnx")
    print("✅ Model loaded successfully!")
    
    # Identify the expected input format
    input_name = session.get_inputs()[0].name
    print(f"Input Name: {input_name}")
    
    # Run a test with dummy data (CIFAR-10 shape)
    dummy_data = np.random.randn(1, 3, 32, 32).astype(np.float32)
    session.run(None, {input_name: dummy_data})
    print("✅ Inference test passed!")
except Exception as e:
    print(f"❌ Verification failed: {e}")