# YOLOV8_optimization_DeepX_NPU-
Designed and implemented optimization pipelines for YOLOv8 to enable high-performance deployment on DeepX Neural Processing Units (NPUs). Applied techniques such as Structured Pruning, quantization,  to reduce latency, memory footprint, and MACs while maintaining detection accuracy.


# Model 

The model was trained using the person class from the COCO Dataset, then pruned and optimized for real-time inference. The final optimized model was deployed and validated on the DeepX M1 NPU chip, ensuring efficient edge-level performance.



## Requirements
Make sure the following dependencies are installed before running the project:

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy


## Approch Used 

### Problem: L2-Norm Pruning Limitation

L2-norm pruning removes filters based on weight magnitude

Assumption: smaller weights → less important filters

### Impact on Small Object Detection

Small objects rely on fine-grained, low-magnitude features

Important filters for small objects may have low L2 norms

Iterative pruning progressively removes these filters

➜ Gradual degradation in small object detection performance


### ✅ Proposed Solution: Taylor-Based Pruning (Taylor Rank)

Rank filters based on their contribution to the loss function

Measures loss sensitivity, not just weight size

If removing a feature map significantly increases loss → it is important

Better preservation of:

* Small object features

L* ocalization-sensitive filters

* Detection robustness



## Results 

Using Taylor Rank method 

Init State : GFLOPs = 82.73 * 2 = 165,4 / Params: 43,691,52 M / mAP = 0,59  

Final state : GFLOPs = 49.05 * 2 = 98.1 /  Params : 18,872,581 M / mAP = 0.56 (mAP drop from baseline : 0.0311 )

Reduction : 40%

SpeeduP : x1.68

Note : Using L2 method 

<img width="800" height="600" alt="pruning_perf_change" src="https://github.com/user-attachments/assets/9e9871fd-cf4c-4b31-bd0e-cf42d7246d16" />
