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

def find_rectangle_with_greatest_height(rectangles, return_all = False):
    if return_all:
        return sorted(rectangles, key=lambda x: x[3])
    rectangle = sorted(rectangles, key=lambda x: x[3])[-1]
    return rectangle

def find_rectangle_with_greatest_width(rectangles, return_all = False):
    if return_all:
        return sorted(rectangles, key=lambda x: x[2])
    rectangle = sorted(rectangles, key=lambda x: x[2])[-1]
    return rectangle

def find_rectangle_with_greatest_area(rectangles, return_all = False):
    if return_all:
        return sorted(rectangles, key=lambda x: x[2] * x[3])
    rectangle = sorted(rectangles, key=lambda x: x[2] * x[3])[-1]
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

def rectangle_to_horizontal_middle(rectangle):
    x, y, w, h = rectangle
    middle = x + w/2
    return middle

def find_central_rectangle(rectangles, rectangle):
    middle = rectangle_to_horizontal_middle(rectangle)
    central_rectangle = sorted(rectangles, key=lambda x: abs(x[0] - middle))[0]
    return central_rectangle

def find_paper_edge(filename):
    image = load_image(filename)
    thresholded_image = threshold_image(image)
    contours = find_contours(thresholded_image)
    rectangles = contours_to_rectangles(contours)
    rectangle= find_rectangle_with_greatest_height(rectangles)
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
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 155, 0), 29)
    plt.ion()
    plt.figure()
    plt.imshow(img)
    plt.show()
    

def crop_with_rectangle(image, rectangle = None, default_height = 600):
    print(image,rectangle)
    if not rectangle: return image[:default_height]
    x, y, w, h = rectangle
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def to_color_image(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def _crop_with_central_rectangle(image, central_rectangle, adjust_x = 0):
    x, y, w, h = central_rectangle
    middle = int(x + w/2) - adjust_x
    print(middle,x, w, adjust_x)
    left_page = image[:, 0:middle]
    right_page = image[:, middle:]
    return left_page, right_page 

def find_two_page_rectangle(ar,hr,wr, image):
    h, w = image.shape
    biggest = ar[-1]
    heighest = hr[-1]
    widest = wr[-1]
    if heighest[3] < 0.9 * h: 
        print('Warning: heighest rectangle is not high enough')
    if widest[2] < 0.9 * w: 
        print('Warning: widest rectangle is not wide enough')
    print('ar',ar[-1])
    print('hr',hr[-1])
    print('wr',wr[-1])
    return wr[-1]

def find_double_paper_edge(filename):
    image = load_image(filename)
    thresholded_image = threshold_image(image)
    contours = find_contours(thresholded_image)
    rectangles = contours_to_rectangles(contours)
    ar= find_rectangle_with_greatest_area(rectangles, return_all = True)
    hr= find_rectangle_with_greatest_height(rectangles, return_all = True)
    wr= find_rectangle_with_greatest_width(rectangles, return_all = True)
    rectangle = find_two_page_rectangle(ar,hr,wr, image)
    rectangle = reduce_rectangle(rectangle, left_inward = 10)
    central_rectangle = find_central_rectangle(rectangles, rectangle)
    return image, rectangle, central_rectangle

def split_image_in_pages(filename):
    image, rectangle, cr = find_double_paper_edge(filename)
    central_rectangle = cr
    adjust_x = 10 + rectangle[0]
    image = crop_image(image, rectangle)
    # image = crop_with_rectangle(image, rectangle)
    return _crop_with_central_rectangle(image, central_rectangle, adjust_x)

def plot_left_right_pages(left_page, right_page):
    plt.ion()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(left_page)
    plt.subplot(1,2,2)
    plt.imshow(right_page)
    plt.show()
    
