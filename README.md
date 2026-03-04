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

## Results 

Using Taylor Rank method to rank the filters ==> 40% reduction

Init State : GFLOPs = 82.73 * 2 = 165,4 / Params: 43,691,52 M / mAP = 0,59  
Final state : GFLOPs = 49.05 * 2 = 98.1 /  Params : 18,872,581 M / mAP = 0.56 (mAP drop from baseline : 0.0311 )
SpeeduP : x1.68

<img width="800" height="600" alt="pruning_perf_change" src="https://github.com/user-attachments/assets/9e9871fd-cf4c-4b31-bd0e-cf42d7246d16" />
