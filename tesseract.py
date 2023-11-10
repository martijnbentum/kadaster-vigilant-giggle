import pytesseract 
from PIL import Image
import image_util
import pickle
from matplotlib import pyplot as plt

header='level,page_num,block_num,par_num,line_num,word_num,left,top,width'
header+=',height,conf,text'
header = header.split(',')

selection = 'left,top,width,height,text'.split(',')

top_line_words = 'NAAM,VOORNAMEN,BEROEP,WOONPLAATS'.split(',')
special_words = 'VRUCHTGEBRUIKER,EIGENAAR,MEDE-EIGENAREN'
special_words = top_line_words + special_words.split(',')



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
    def __init__(self, filename):
        '''filename: path to image file
        index: height index to crop image at only the top part is needed
        '''
        self.filename = filename
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
        self.image, self.rectangle = image_util.find_paper_edge(self.filename)
        self.cropped_image=image_util.crop_image(self.image, self.rectangle)
        self.thresholded_image=image_util.threshold_image(self.cropped_image)

    def _set_infos(self):
        self.infos = []
        self.specials = []
        self.top = []
        for index in range(200,900,100):
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
        return self.filename.split('/')[-2]
        
    def show_thresholded(self, index = 600):
        '''show the thresholded image'''
        plt.ion()
        plt.clf()
        plt.imshow(self.thresholded_image[:index])
        plt.show()        

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
    
    


