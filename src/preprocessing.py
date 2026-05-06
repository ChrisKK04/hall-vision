import cv2
import matplotlib.pyplot as plt

def preprocess(img_path, digit, threshold=255, show=False):
    # load image as grayscale
    image_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # resize to 28 x 28
    resize = cv2.resize(image_gray, (28, 28))    
    # set pixels above threshold to white
    resize[resize > threshold] = 255
    # set intensity range to 0 - 1
    normalized = resize / 255.0
    # change pixels black -> white and white -> black
    flipped = abs(normalized - 1)

    # print the preprocessed
    if show:
        plt.imshow(flipped, cmap="gray")
        plt.show()

    # transform the image into a 1D vector for inference
    output = flipped.reshape(784, 1)

    return (output, digit)

if __name__ == "__main__":
    input = preprocess(img_path="images_test/zero.png", digit=0, threshold=255, show=True)