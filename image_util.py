import cv2
from matplotlib import pyplot as plt
import copy


def load_image(filename):
    image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    return image

def threshold_image(image):
    _, threshold = cv2.threshold(image, 0, 255, 
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold

def find_contours(thresholded_image):
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_TREE, 
        cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contours_to_rectangles(contours):
    rectangles = []
    for contour in contours:
        rectangle = cv2.boundingRect(contour)
        rectangles.append(rectangle)
    return rectangles

def find_rectangle_with_greatest_height(rectangles):
    rectangle = sorted(rectangles, key=lambda x: x[3])[-1]
    return rectangle

def reduce_rectangle(rectangle, left_inward = 10, top_inward = 50, 
    right_inward = 20, bottom_inward = 150):
    x, y, w, h = rectangle
    x += left_inward
    y += top_inward
    w -= (left_inward + right_inward)
    h -= (top_inward + bottom_inward)
    rectangle = x,y,w,h
    return rectangle

def find_paper_edge(filename):
    image = load_image(filename)
    thresholded_image = threshold_image(image)
    contours = find_contours(thresholded_image)
    rectangles = contours_to_rectangles(contours)
    rectangle = find_rectangle_with_greatest_height(rectangles)
    rectangle = reduce_rectangle(rectangle)
    return image, rectangle

def crop_image(image, rectangle):
    x, y, w, h = rectangle
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def show_image_with_rectangle(image, rectangle):
    img = copy.copy(image)
    print(rectangle)
    x, y, w, h = rectangle
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 9)
    plt.ion()
    plt.clf()
    plt.imshow(img)
    plt.show()
    
    
