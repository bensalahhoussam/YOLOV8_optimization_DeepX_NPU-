# YOLOV8_optimization_DeepX_NPU-
Designed and implemented optimization pipelines for YOLOv8 to enable high-performance deployment on DeepX Neural Processing Units (NPUs). Applied techniques such as pruning, quantization, and computational graph optimization to reduce latency, memory footprint, and MACs while maintaining detection accuracy.


## Requirements
ake sure the following dependencies are installed before running the project:

- Python 3.8+
- Qt6
- OpenCV
- Ultralytics (YOLOv8)
- NumPy

## Results 

Using L2 method to rank the filters ==> 71% reduction

Init State : FLOPs = 82 ,73 * 2 = 165,4 / Params: 43,691,52 M / mAP = 0,65  
Final state : FLOPs = 24* 32 = 48.64 /  Params : 12,496,550 M / mAP = 0.67 (mAP drop from baseline : -0.0239 )


<img width="761" height="572" alt="image" src="https://github.com/user-attachments/assets/39565350-bc5a-4082-9bac-6857ffb3b6fa" />

