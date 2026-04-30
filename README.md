# Local Statistics & NumPy Optimization in Computer Vision

##  Project Overview
This project is a performance-focused tool developed for the **Computer Vision** course. It calculates local statistical data—specifically **Mean** and **Variance**—for image channels using a block-wise approach. 

The primary goal of this project is to demonstrate **algorithmic optimization** by comparing standard Python nested loops against vectorized **NumPy** operations. This type of optimization is critical in automotive software engineering, where real-time processing of camera feeds is required for driver assistance systems.

##  Technical Features
*   **Block-wise Analysis:** Divides images into $16 \times 16$ blocks to extract local texture and intensity features.
*   **Performance Benchmarking:** Compares execution times between $O(n^2)$ loop-based processing and optimized NumPy vectorization[.
*   **Multi-Channel Processing:** Analyzes different color spaces, including **Red (RGB)** and **Hue (HSV)**, to understand how statistical variance shifts across channels.
*   **Dynamic Normalization:** Implements a Min-Max normalization algorithm to convert raw variance data into viewable 8-bit images (0-255)[.

##  Technologies Used
*   **Python 3**
*   **OpenCV (cv2):** For image loading, color space conversion, and output generation.
*   **NumPy:** For high-performance array manipulation and vectorized mathematical operations[.
*   **Time Module:** For precise execution latency measurement.

##  Performance Comparison
In this project, the `process_numpy` function utilizes **array broadcasting** and **reshaping**. This allows the CPU to process the entire image grid simultaneously, significantly reducing the "NumPy Time" compared to the "Loop Time".

## 📂 How to Run
* Clone this repo to your local environment 
    **git clone https://github.com/nirmalajay/Automotive_CV_Image_stats.git
    **change to working directory
* pip install requirements.txt 
* python3 main.py 