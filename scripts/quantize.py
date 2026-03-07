import os
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "../models")

input_model = os.path.join(MODELS_DIR, "model.onnx")
preproc_model = os.path.join(MODELS_DIR, "model_preproc.onnx")
output_model = os.path.join(MODELS_DIR, "model_int8.onnx")

# pre-process to fix shape inference issues
quant_pre_process(input_model, preproc_model)

quantize_dynamic(
    model_input=preproc_model,
    model_output=output_model,
    weight_type=QuantType.QInt8,
)

print("Quantized model saved to models/model_int8.onnx")