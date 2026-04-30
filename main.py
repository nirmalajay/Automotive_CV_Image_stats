import cv2
import numpy as np
import time
import os

IMAGE_PATH = "tulips.jpg"
BLOCK_SIZE = 16
OUTPUT_DIR = "TEST_OUTPUT"


def process_loops(channel_data, block_size):

    h, w = channel_data.shape
    mean_f = np.zeros((h, w), dtype=np.float64)
    var_f = np.zeros((h, w), dtype=np.float64)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            
            block_sum = 0.0
            pixel_count = 0 
            
            #mean
            for row in range(y, y + block_size):
                for col in range(x, x + block_size):
                    if row < h and col < w:
                        block_sum += channel_data[row, col]
                        pixel_count += 1
                        
            mu = block_sum / pixel_count

            #variance
            sq_diff_sum = 0.0
            for row in range(y, y + block_size):
                for col in range(x, x + block_size):
                    if row < h and col < w:
                        sq_diff_sum += (channel_data[row, col] - mu) ** 2
                        
            variance = sq_diff_sum / pixel_count


            for row in range(y, y + block_size):
                for col in range(x, x + block_size):
                    if row < h and col < w:
                        mean_f[row, col] = mu
                        var_f[row, col] = variance

    return mean_f, var_f


def process_numpy(channel_data, block_size):

    h, w = channel_data.shape
    
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded_img = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode='constant')
    
    ph, pw = padded_img.shape
    grid_h = ph // block_size
    grid_w = pw // block_size
    num_pixels = block_size * block_size

    blocks = padded_img.reshape(grid_h, block_size, grid_w, block_size)

    block_means = np.sum(blocks, axis=(1, 3), dtype=np.float64) / num_pixels

    blocks_float = blocks.astype(np.float64)
    diff        = blocks_float - block_means[:, np.newaxis, :, np.newaxis]
    block_vars   = np.sum(diff ** 2, axis=(1, 3)) / num_pixels

    mean_full = np.repeat(np.repeat(block_means, block_size, axis=0), block_size, axis=1)
    var_full = np.repeat(np.repeat(block_vars, block_size, axis=0), block_size, axis=1)

    return mean_full[:h, :w], var_full[:h, :w]


def Normalizing(image_matrix, filename):

    min_val = np.min(image_matrix)
    max_val = np.max(image_matrix)
    
    if max_val == min_val:
        normalized = np.zeros_like(image_matrix, dtype=np.uint8)
    else:
        normalized = ((image_matrix - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), normalized)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    img_bgr = cv2.imread(IMAGE_PATH)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    channels = {
        "Red": img_bgr[:, :, 2],
        #"Green": img_bgr[:, :, 1],
        #"Blue": img_bgr[:, :, 0],
        #"Gray": img_gray,
        "Hue": img_hsv[:, :, 0],
        #"Saturation": img_hsv[:, :, 1],
        #"Value": img_hsv[:, :, 2]
    }

    print(f"Input image: '{IMAGE_PATH}' | Block Size: {BLOCK_SIZE}x{BLOCK_SIZE}")
    print("-" * 50)

    for name, channel_data in channels.items():
        print(f"Channel: {name}")

        #loop
        start_loop = time.perf_counter()
        mean_loop, var_loop = process_loops(channel_data, BLOCK_SIZE)
        end_loop = time.perf_counter()
        loop_time = end_loop - start_loop

        #numpy
        start_np = time.perf_counter()
        mean_np, var_np = process_numpy(channel_data, BLOCK_SIZE)
        end_np = time.perf_counter()
        np_time = end_np - start_np

        print(f"  Loop Time  : {loop_time:.4f}s")
        print(f"  NumPy Time : {np_time:.4f}s")
        

        mean_final = mean_np.astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_Mean.png"), mean_final)

        Normalizing(var_np, f"{name}_Variance.png")

    print("-" * 50)
    print(f"Completed. Results stored in '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()