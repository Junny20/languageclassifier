import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import os

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "../models")

class LanguageClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, 3)
        self.conv2 = nn.Conv1d(128, 128, 3)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input):
        input = self.embedding(input)
        input = input.permute(0, 2, 1)
        input = torch.relu(self.conv1(input))
        input = torch.relu(self.conv2(input))
        input = input.max(2).values
        input = torch.relu(self.fc1(input))
        input = self.dropout(input)
        input = self.fc2(input)
        return input
    
cpp_model = torch.jit.load(os.path.join(MODELS_DIR, "model.pt")) # loads torchScript
py_model = LanguageClassifier(256, 32, 25) # builds identical architecture in python
py_model.load_state_dict(cpp_model.state_dict()) # extracts weights from torchScript
py_model.eval()

dummy_input = torch.randint(0, 256, (1, 200), dtype=torch.long) # shape: {1, 200}

onnx_program = torch.onnx.export(
    py_model,
    (dummy_input,),
    dynamo=True
)

onnx_program.save(os.path.join(MODELS_DIR, "model.onnx"))

print("Exported to models/model.onnx")

# RUN INFERENCE! (Validating model through both pytorch and onnxruntime)
with torch.no_grad():
    pytorch_output = py_model(dummy_input).numpy()

session = ort.InferenceSession(os.path.join(MODELS_DIR, "model.onnx"))

# print("Inputs:", [i.name for i in session.get_inputs()])
# print("Outputs:", [o.name for o in session.get_outputs()])

# session.run returns a list of tensors, one for each output - we collect the first one
onnx_output = session.run(["linear_1"], {"input": dummy_input.numpy()})[0] 

max_diff = np.max(np.abs(pytorch_output - onnx_output))
print(f"Maximum difference between PyTorch and ONNX model outputs: {max_diff}")
if max_diff < 1e-4:
    print("Validation passed!")
else:
    print("Validation failed... outputs don't match :(")