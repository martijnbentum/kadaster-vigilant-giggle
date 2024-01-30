import copy
import cv2
import pytesseract 
from PIL import Image
import image_util
import locations
import pickle
from matplotlib import pyplot as plt
import numpy as np
import os
import viti

header='level,page_num,block_num,par_num,line_num,word_num,left,top,width'
header+=',height,conf,text'
header = header.split(',')

selection = 'left,top,width,height,text'.split(',')

top_line_words = 'NAAM,VOORNAMEN,BEROEP,WOONPLAATS'.split(',')
special_words = 'VRUCHTGEBRUIKER,EIGENAAR,MEDE-EIGENAREN'
special_words = top_line_words + special_words.split(',')


def make_viti_output_filename(data):
    x = data
    filename = locations.viti_outputs + x.scan_type + '_' + x.name + '.pickle'
    return filename

def save_viti_outputs(data, processor,  model):
    outputs = viti.handle_data(data, processor, model)
    filename = make_viti_output_filename(data)
    viti.save_outputs(outputs, filename)
    return outputs


def save_all_viti_outputs(processor = None, model = None):
    if not processor:processor = viti.load_processor()
    if not model: model = viti.load_model()
    for filename in locations.all_files:
        print(filename)
        data = Data(filename)
        save_viti_outputs(data, processor, model)



def load_image(filename):
    img = Image.open(filename)
    return img

def get_text(img, language='nld'):
    text = pytesseract.image_to_string(img,lang = language)
    return text

def get_data(img, language='nld'):
    data = pytesseract.image_to_data(img, lang=language)
    return data

class Data:
    '''data structure for tesseract data'''
    def __init__(self, filename, indices = range(200,900,100), crop_image=True,
        output_directory = ''):
        '''filename: path to image file
        index: height index to crop image at only the top part is needed
        '''
        self.filename = filename
        self.indices = indices
        self.crop_image = crop_image
        self.output_directory = output_directory
        self.name = filename.split('/')[-1].split('.')[0]
        self.header = header
        self._handle_image()
        self._set_infos()

    def __repr__(self):
        m = ', '.join([x.text for x in self.specials])
        return m

    def _handle_image(self):
        '''load the scanned image, crop it, and threshold it
        the scan is cropped to remove the black border 
        the scan is thresholded to enhance the text
        '''
        if self.crop_image:
            self.image, self.rectangle = image_util.find_paper_edge(self.filename)
            width = self.rectangle[2]
            if width < 2000: 
                self.cropped_image = copy.copy(self.image)
            else:
                self.cropped_image=image_util.crop_image(self.image, 
                self.rectangle)
        else:
            self.image = image_util.load_image(self.filename)
            self.cropped_image = copy.copy(self.image)
        self.thresholded_image=image_util.threshold_image(self.cropped_image)

    def _set_infos(self):
        self.infos = []
        self.specials = []
        self.top = []
        for index in self.indices:
            info = Info(self, index)
            if len(info.lines) > 0:
                self.infos.append( info )
            for top_line in info.top:
                if top_line not in self.top:
                    self.top.append(top_line)
            for special_line in info.specials:
                if special_line not in self.specials:
                    self.specials.append(special_line)
        d = {x.text.strip(' ,.').lower(): x for x in self.specials}
        self.special_dict = d
        d = {x.text.strip(' ,.').lower(): x for x in self.top}
        self.top_dict = d

    @property
    def scan_type(self):
        '''returns the type of scan based on the filename
        types: with_stamp, no_stamp, with_note, no_note
        '''
        if 'split_pages' in self.filename:
            return 'unknown'
        return self.filename.split('/')[-2]
        
    def show_thresholded(self, index = 600):
        '''show the thresholded image'''
        plt.ion()
        plt.clf()
        plt.imshow(self.thresholded_image[:index])
        plt.show()        

    @property
    def top_rectangle(self):
        '''returns the rectangle containing owner information
        this is the area a stamp can occur
        '''
        if hasattr(self, '_top_rectangle'): return self._top_rectangle
        f = top_word_to_top_rectangle
        self._top_rectangles = [f(v,k) for k,v in self.top_dict.items()]
        self._top_rectangle = median_rectangle(self._top_rectangles)
        return self._top_rectangle

    @property
    def name_rectangle(self):
        '''returns the rectangle containing the name of the owner'''
        if 'eigenaar' not in self.special_dict: return None
        if hasattr(self, '_name_rectangle'): return self._name_rectangle
        eigenaar_line = self.special_dict['eigenaar']
        temp = to_name_rectangle(eigenaar_line, self.top_rectangle)
        self._name_rectangle = temp
        return self._name_rectangle

    @property
    def name_image(self):
        '''returns the image containing the name of the owner'''
        if hasattr(self, '_name_image'): return self._name_image
        self._name_image = image_util.crop_with_rectangle(self.cropped_image, 
            self.name_rectangle)
        return self._name_image

    @property
    def top_image(self):
        '''returns the image containing the top part of the scan'''
        if hasattr(self, '_top_image'): return self._top_image
        self._top_image = image_util.crop_with_rectangle(self.cropped_image, 
            self.top_rectangle)
        return self._top_image

    def save_top_image(self):
        if self.output_directory: path = self.output_directory
        else: path = locations.cropped_top + self.scan_type + '/'
        if not os.path.isdir(path): os.mkdir(path)
        f = path + self.name + '.jpg'
        cv2.imwrite(f, self.top_image)

    def save_name_image(self):
        if self.output_directory: path = self.output_directory
        else:path = locations.cropped_name + self.scan_type + '/'
        if not os.path.isdir(path): os.mkdir(path)
        f = path + self.name + '.jpg'
        cv2.imwrite(f, self.name_image)

    def save_name_and_top_image(self):
        self.save_name_image()
        self.save_top_image()

    

class Info:
    def __init__(self, data, index):
        self.data = data
        self.index = index
        self._set_info()


    def _set_info(self):
        index = self.index
        self.dataset = get_data(self.data.thresholded_image[:index])
        self.dataset = [x.split('\t') for x in self.dataset.split('\n')[1:]]
        self.lines = []
        for line in self.dataset:
            line = Line(line, self)
            if line.ok:
                self.lines.append(line)
        self.specials = [line for line in self.lines if line.is_special]
        self.top = [line for line in self.lines if line.is_top]
        d = {x.text.strip(' ,.').lower(): x for x in self.specials}
        self.special_dict = d
        d = {x.text.strip(' ,.').lower(): x for x in self.top}
        self.top_dict = d


class Line:
    def __init__(self, line, info):
        self.line = line
        self.info = info
        self.data = info.data
        self.ok = True
        if len(line) == len(self.data.header):
            self._set_info()
        else: self.ok = False

    def __repr__(self):
        return self.text

    def __eq__(self, other):
        if not type(self) == type(other): return False
        return self.text == other.text

    def _set_info(self):
        for i, attr_name in enumerate(self.data.header):
            value = self.line[i] 
            if attr_name == 'text': pass
            elif value.isdigit(): value = int(value)
            setattr(self, attr_name, value)
        if self.text == '':
            self.ok = False
            
    @property
    def is_special(self):
        if not self.ok: return False
        if self.text.strip(' ,.') in special_words:
            return True
        return False

    @property
    def is_top(self):
        if not self.ok: return False
        if self.text.strip(' ,.') in top_line_words:
            return True
        return False

    @property
    def rectangle(self):
        return (self.left, self.top, self.width, self.height)

    @property
    def x(self):
        return self.left

    @property
    def y(self):
        return self.top

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def top_rectangle(self):
        return top_word_to_top_rectangle(self, self.text.strip(' ,.'))
    
    
def top_word_to_top_rectangle(line, word):
    ''' converts ocr location information from a specific word to 
    the rectangle of interest (containing the top part of the scan
    '''
    x,y,w,h = line.rectangle
    if h > 20: h = 20
    left,top = 0,0
    if word == 'naam':
        left = int(x - w * 1.5)
        top = int(y - h * 2.5)
        bottom = int(y + h * 31)
        right = int(x + w * 8)
    elif word == 'voornamen':
        left = int(x - w * 2.7)
        top = int(y - h * 2.5)
        bottom = int(y + h * 32)
        right = int(x + w * 4.9)
    elif word == 'beroep':
        left = int(x - w * 8)
        top = int(y - h * 2.5)
        bottom = int(y + h * 31.5)
        right = int(x + w * 5)
    elif word == 'woonplaats':
        left = int(x - w * 5.7)
        top = int(y - h * 2.5)
        bottom = int(y + h * 31)
        right = int(x + w * 1.3)
    else: return None
    if left < 0: left = 0
    if top < 0: top = 0
    rectangle = (left, top, right-left, bottom-top)
    return rectangle

def median_rectangle(rectangles):
    if len(rectangles) == 0: return None
    if len(rectangles) == 1: return rectangles[0]
    array = np.array(rectangles)
    median = np.median(array, axis=0)
    return list(map(int, median))
    
def mean_rectangle(rectangles):
    if len(rectangles) == 0: return None
    if len(rectangles) == 1: return rectangles[0]
    array = np.array(rectangles)
    mean = np.mean(array, axis=0)
    return list(map(int, mean))
    
def to_name_rectangle(eigenaar_line = None, top_rectangle = None):
    ''' converts ocr location information from ocr line with 'eigenaar' to 
    the rectangle of interest (containing the person information)
    '''
    el_rectangle, tr_rectangle = None, None 
    if eigenaar_line:
        x,y,w,h = eigenaar_line.rectangle
        if h > 21: h = 21
        left = int(x + w * 2.9)
        top = int(y - h * 4.5)
        bottom = int(y + h * 3)
        right = int(x + w * 17.4)
        if top < 0: top = 0
        el_rectangle = (left, top, right-left, bottom-top)
    if top_rectangle:
        x,y,w,h = top_rectangle
        left = x
        top = int(y + h *.18)
        bottom = int(y + h * .4)
        right = x + w
        tr_rectangle = (left, top, right-left, bottom-top) 
    if el_rectangle and tr_rectangle:
        return mean_rectangle([el_rectangle, tr_rectangle])
    elif el_rectangle: return el_rectangle
    elif tr_rectangle: return tr_rectangle
    return None

def _log_line(d):
    line = []
    line.append(d.filename)
    line.append(d.scan_type)
    line.append(str(bool(d.top_rectangle)))
    line.append(str(bool(d.name_rectangle)))
    line.append(','.join(d.special_dict.keys()))
    line.append(','.join(d.top_dict.keys()))
    return '\t'.join(line)

def make_all_cropped_images():
    filenames = locations.all_files
    log = []
    datas = []
    for i,filename in enumerate(filenames):
        print(i,filename,len(filenames))
        data = Data(filename)
        data.save_name_and_top_image()
        log.append(_log_line(data))
        datas.append(data)
    with open('../log.tsv', 'w') as f:
        f.write('\n'.join(log))
    return datas

