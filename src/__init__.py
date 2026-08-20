import sys 
import cv2 as cv
import numpy as np
from pathlib import Path

# folder = Path('data/SROIE2019/train/img')
# image_paths = [p for p in folder.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
# num_images = len(image_paths)

# print(folder)
# # print(image_paths)
# print(num_images)

def image_batch_generator(folder_path, batch_size=32, target_size=(224, 224)):
    """Yields batches of images and labels sequentially without clogging RAM."""
    folder = Path(folder_path)
    # Find all common image formats
    image_paths = [p for p in folder.rglob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    num_images = len(image_paths)
    
    while True:
        for i in range(0, num_images, batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                # OpenCV loads in BGR format by default
                img = cv.imread(str(path))
                if img is not None:
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    bilateralFilter = cv.bilateralFilter(gray, 9, 75, 75)
                    # edges = cv.Canny(bilateralFilter, 100, 200)
                    batch_images.append(bilateralFilter)
                    cv.imshow("My image", bilateralFilter)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
            yield np.array(batch_images)

data_gen = image_batch_generator("data/SROIE2019/train/img", batch_size=5)
first_batch = next(data_gen)
