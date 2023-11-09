import pytesseract 
from PIL import Image
import image_util

header='level,page_num,block_num,par_num,line_num,word_num,left,top,width'
header+=',height,conf,text'
header = header.split(',')

selection = 'left,top,width,height,text'.split(',')

special_words = 'NAAM,VOORNAMEN,BEROEP,WOONPLAATS,VRUCHTGEBRUIKER,'
special_words += 'EIGENAAR,MEDE-EIGENAREN'
special_words = special_words.split(',')



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
    def __init__(self, filename, index = 600):
        self.filename = filename
        self.header = header
        self._handle_image()
        self.data = get_data(self.thresholded_image[:600])
        self.data = [x.split('\t') for x in self.data.split('\n')[1:]]
        self._set_info()

    def __repr__(self):
        m = self.filename.split('/')[-1] 
        m += ', '.join([x.text for x in self.specials])
        return m

    def _handle_image(self):
        self.image, self.rectangle = image_util.find_paper_edge(self.filename)
        self.cropped_image = image_util.crop_image(self.image, self.rectangle)
        self.thresholded_image = image_util.threshold_image(self.cropped_image)

    def _set_info(self):
        self.lines = []
        for line in self.data:
            line = Line(line, self)
            if line.ok:
                self.lines.append(line)
        self.specials = [line for line in self.lines if line.is_special]
        


class Line:
    def __init__(self, line, data):
        self.line = line
        self.data = data
        self.ok = True
        if len(line) == len(self.data.header):
            self._set_info()
        else: self.ok = False

    def __repr__(self):
        return self.text

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
    
    


