# YOLOV8_optimization_DeepX_NPU_M1
Designed and implemented optimization pipelines for YOLOv8 to enable high-performance deployment on DeepX Neural Processing Units (NPUs). Applied techniques such as Structured Pruning, quantization,  to reduce latency, memory footprint, and MACs while maintaining detection accuracy.


# Model 

The model was trained using the person class from the COCO Dataset, then pruned and optimized for real-time inference. The final optimized model was deployed and validated on the DeepX M1 NPU chip, ensuring efficient edge-level performance.

You can download the trained models below:

- **Full Size Model**  
  👉 [Download here](https://drive.google.com/file/d/147LdITmBO8U7Ph6797X0qFvYhAiovxSk/view?usp=sharing)

- **Pruned Model**  
  👉 [Download here](https://drive.google.com/file/d/1B6cFLSjGSeUOtiKna4kfbaPl9yrbGeuV/view?usp=drive_link)
  
## Requirements
Make sure the following dependencies are installed before running the project:

- Python 3.8+
- OpenCV
- Ultralytics (YOLOv8)
- NumPy


## 📌 Approach Used 

### Problem: L2-Norm Pruning Limitation

L2-norm pruning removes filters based solely on weight magnitude.

Assumption: smaller weights → less important filters
Smaller weights → Less important filters


### 🎯 Impact on Small Object Detection

* Small objects depend on fine-grained and subtle feature representations

* Filters capturing these details may have low L2 norms

* Magnitude-based pruning may incorrectly remove these filters

* Iterative pruning progressively eliminates informative features

➜ Result: Gradual degradation in small object detection performance


### ✅ Proposed Solution: Taylor-Based Pruning (Taylor Rank)

Instead of relying on weight magnitude, we:

* Rank filters based on their contribution to the loss function

* Estimate importance using loss sensitivity (first-order Taylor approximation)

* Preserve filters whose removal significantly increases the loss

### 🚀 Benefits of Taylor-Based Pruning

* Better preservation of:

  * Small object feature representations

  * Localization-sensitive filters

  * Detection robustness

* Maintains accuracy while enabling structured model compression

* More reliable for real-time deployment scenarios


## Results 

Init State : GFLOPs = 82.73 * 2 = 165,4 / Params: 43,691,52 M / mAP = 0,59  

Final state : GFLOPs = 49.05 * 2 = 98.1 /  Params : 18,872,581 M / mAP = 0.56 (mAP drop from baseline : 0.0311 )

Reduction : 40%

SpeeduP : x1.68

<img width="800" height="600" alt="pruning_perf_change" src="https://github.com/user-attachments/assets/9e9871fd-cf4c-4b31-bd0e-cf42d7246d16" />



| Original Model  | Pruned Model |
|---------|---------|
| [![Video 1](thumbnail1.png)]() | [![Video 2](thumbnail2.png)](https://link_to_video2.com) |

