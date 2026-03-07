import os
import time
import numpy as np
import onnxruntime as ort

SCRIPTS_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(SCRIPTS_DIR, "../models")

def benchmark(model_path, label, n=1000):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randint(0, 256, (1, 200)).astype(np.int64)

    # warmup :>
    for _ in range(50):
        session.run(None, {input_name: dummy_input})
    
    times = []
    for _ in range(n):
        start = time.perf_counter_ns() # measured in ns
        session.run(None, {input_name: dummy_input})
        end = time.perf_counter_ns()
        times.append((end - start) / 1_000_000)
    
    times = np.array(times)
    print(f"\n{label}")
    print(f"  Avg latency:  {times.mean():.3f} ms")
    print(f"  p50 latency:  {np.percentile(times, 50):.3f} ms")
    print(f"  p99 latency:  {np.percentile(times, 99):.3f} ms")
    print(f"  Model size:   {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    return times.mean()

fp32_avg = benchmark(os.path.join(MODELS_DIR, "model.onnx"), "FP32")
int8_avg = benchmark(os.path.join(MODELS_DIR, "model_int8.onnx"), "INT8")

print(f"\nSpeedup: {fp32_avg / int8_avg:.2f}x")
print(f"Size reduction: {os.path.getsize(os.path.join(MODELS_DIR, 'model.onnx')) / os.path.getsize(os.path.join(MODELS_DIR, 'model_int8.onnx')):.2f}x")