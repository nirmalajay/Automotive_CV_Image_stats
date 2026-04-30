# Local Statistics & NumPy Optimization in Computer Vision

##  Project Overview
This project is a performance-focused tool developed for the **Computer Vision** course[cite: 1]. It calculates local statistical data—specifically **Mean** and **Variance**—for image channels using a block-wise approach. 

The primary goal of this project is to demonstrate **algorithmic optimization** by comparing standard Python nested loops against vectorized **NumPy** operations. This type of optimization is critical in automotive software engineering, where real-time processing of camera feeds is required for driver assistance systems.

##  Technical Features
*   **Block-wise Analysis:** Divides images into $16 \times 16$ blocks to extract local texture and intensity features[cite: 2].
*   **Performance Benchmarking:** Compares execution times between $O(n^2)$ loop-based processing and optimized NumPy vectorization[cite: 2].
*   **Multi-Channel Processing:** Analyzes different color spaces, including **Red (RGB)** and **Hue (HSV)**, to understand how statistical variance shifts across channels[cite: 2].
*   **Dynamic Normalization:** Implements a Min-Max normalization algorithm to convert raw variance data into viewable 8-bit images (0-255)[cite: 2].

##  Technologies Used
*   **Python 3**[cite: 1, 2]
*   **OpenCV (cv2):** For image loading, color space conversion, and output generation[cite: 2].
*   **NumPy:** For high-performance array manipulation and vectorized mathematical operations[cite: 2].
*   **Time Module:** For precise execution latency measurement[cite: 2].

##  Performance Comparison
In this project, the `process_numpy` function utilizes **array broadcasting** and **reshaping**[cite: 2]. This allows the CPU to process the entire image grid simultaneously, significantly reducing the "NumPy Time" compared to the "Loop Time"[cite: 2].

## 📂 How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install opencv-python numpy