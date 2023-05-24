import os
import sys
import csv
import numpy as np
from PIL import Image
from keras.models import load_model

model = load_model('final_model.h5')

by_merge_map = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B',
    12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M',
    23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q',
    45: 'r', 46: 't'
}

def process_image_directory(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(filename)

    if not images:
        print("No images found in the directory.")
        return

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Character', 'Image Path', 'Predicted Label'])
        for image in images:
            image_path = os.path.join(directory, image)

            # Load and preprocess the image
            img = Image.open(image_path).convert('L')
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = img_array.reshape(1, 28, 28, 1)
            img_array = img_array.astype('float32') / 255.0  # Convert to binary format

            # Perform inference using the trained model
            prediction = model.predict(img_array)
            predicted_label = np.argmax(prediction)
            character = by_merge_map[predicted_label]

            writer.writerow([character, image_path, character])

    print("Output saved to output.csv")

image_directory = r"C:\Python Project\Intership\input"
process_image_directory(image_directory)
