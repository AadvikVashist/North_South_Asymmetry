import os
import numpy as np
def analyze_image_for_brightness(image):
    return np.mean(image)
def get_brightness(file,folder_path):
    cubes = [a[0] for a in file]
    # iterate through all files in the folder
    results = file[0]; results.append("brightness"); results.append(brightness)
    walked = [os.path.join(folder_path,f).replace("\\","/") for f in cubes if any((f.endswith(".jpg"),f.endswith(".png"), f.endswith(".tif")))]
    for index, filename in enumerate(walked):
        image_path = filename
        brightness = analyze_image_for_brightness(image_path)
        row = file[index]; row.append(brightness)
        results.append(row)
    return results
if __name__ == "__main__":
    folder_path = "D:/Nantes/"
    results = get_brightness(folder_path)
    print(results)