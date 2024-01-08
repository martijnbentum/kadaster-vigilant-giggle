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

def _split_pages(image, middle):
    left_page = image[:, 0:middle]
    right_page = image[:, middle:]
    return left_page, right_page 

def find_left_right_edge(hr, middle):
    lefts = [x for x in hr if x[0] < (middle - middle*.75)]
    rights= [x for x in hr if x[0] > (middle - middle*.75)]
    left = find_rectangle_with_greatest_height(lefts)
    right= find_rectangle_with_greatest_height(rights)
    left_edge = left[0] 
    right_edge = right[0] + right[2]
    return left_edge, right_edge
    
def find_top_bottom_edge(wr, middle):
    tops = [x for x in wr if x[0] < (middle - middle*.75)]
    bottoms = [x for x in wr if x[0] < (middle - middle*.75)]
    top = find_rectangle_with_greatest_height(tops)
    bottom = find_rectangle_with_greatest_height(bottoms)
    top_edge = top[1]
    bottom_edge = bottom[1] + bottom[3]
    return top_edge, bottom_edge

def find_middle_with_rectangles(rectangles, rectangle, left_inward):
    rectangle_middle = rectangle_to_horizontal_middle(rectangle)
    central_rectangle = find_central_rectangle(rectangles, rectangle)
    x, y, w, h = central_rectangle
    adjust_x = left_inward + rectangle[0]
    middle = int(x + w/2) - adjust_x
    if abs(middle - rectangle_middle) > (rectangle_middle *.1):
        print(x,y,w,h,middle)
        print('middle found based on contours is not close to middle of image')
        print('returning middle of two page rectangle')
        middle = int(rectangle_middle) - adjust_x 
        print(x,y,w,h,middle)
    return middle

def find_two_page_rectangle(hr, wr, image):
    image_height, image_width = image.shape
    horizontal_middle,vertical_middle  = image_width / 2, image_height / 2
    heighest= hr[-1]
    widest= wr[-1]
    x, y, w, h = heighest
    if h < image_height * .75: 
        print('hight to small')
        top_edge, bottom_edge = find_top_bottom_edge(wr, vertical_middle)
        y = top_edge
        h = bottom_edge - top_edge
    if  w < image_width * .75: 
        print('width to small')
        left_edge, right_edge = find_left_right_edge(hr, horizontal_middle)
        x = left_edge
        w = right_edge - left_edge
    return x, y, w, h

def find_double_paper_edge(filename, left_inward = 10):
    image = load_image(filename)
    thresholded_image = threshold_image(image)
    contours = find_contours(thresholded_image)
    rectangles = contours_to_rectangles(contours)
    hr= find_rectangle_with_greatest_height(rectangles, return_all = True)
    wr= find_rectangle_with_greatest_width(rectangles, return_all = True)
    rectangle = find_two_page_rectangle(hr,wr, image)
    rectangle = reduce_rectangle(rectangle, left_inward = left_inward)
    middle = find_middle_with_rectangles(rectangles, rectangle, left_inward)
    return image, rectangle, middle

def split_image_in_pages(filename):
    image, rectangle, middle = find_double_paper_edge(filename)
    image = crop_image(image, rectangle)
    return _split_pages(image, middle)

def plot_left_right_pages(left_page, right_page):
    plt.ion()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(left_page)
    plt.subplot(1,2,2)
    plt.imshow(right_page)
    plt.show()
    
