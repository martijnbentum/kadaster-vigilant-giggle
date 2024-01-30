import glob
import image_util
import locations
import os
from progressbar import progressbar
import tesseract
import viti

def load_haarlem_files():
    fn = glob.glob(locations.haarlem + '**/*.jpg', recursive=True)
    return fn

def load_haarlem_pages():
    fn = glob.glob(locations.haarlem_split_pages + '*.jpg')
    return fn

def load_haarlem_top_images():
    fn = glob.glob(locations.haarlem_top_images + '*.jpg')
    return fn

def load_haarlem_top_images():
    fn = glob.glob(locations.haarlem_viti_outputs + '*.jpg')
    return fn

def split_pages(f):
    left_page, right_page = image_util.split_image_in_pages(f)
    return left_page, right_page
    
def make_name(f):
    name = f.replace(locations.haarlem,'').replace('.jpg','')
    name = name.replace('/','_')
    return name

def _make_page_names(filename):
    name = make_name(filename)
    left_name = locations.haarlem_split_pages + name + '_left.jpg'
    right_name = locations.haarlem_split_pages + name + '_right.jpg'
    return left_name, right_name

def save_pages(filename):
    left_page, right_page = split_pages(filename)
    left_name, right_name = _make_page_names(filename)
    image_util.save_image( left_name , left_page)
    image_util.save_image( right_name , right_page)
    return left_page, right_page

    
def save_all_pages():
    files = load_haarlem_files()
    for f in progressbar(files):
        left_name, right_name = _make_page_names(f)
        if not os.path.isfile(left_name) or not os.path.isfile(right_name):
            save_pages(f)

def make_all_top_images_haarlem(filenames = None):
    if not filenames: filenames = load_haarlem_pages() 
    datas = []
    for i,filename in enumerate(progressbar(filenames)):
        output_filename = locations.haarlem_top_images + filename.split('/')[-1]
        if os.path.isfile(output_filename): continue
        data = tesseract.Data(filename, crop_image = False, 
            output_directory = locations.haarlem_top_images)
        data.save_top_image()
        log = tesseract._log_line(data)
        with open(locations.haarlem_base + 'log.tsv', 'a') as f:
            f.write(log + '\n')

def make_all_viti_outputs(filenames = None, processor = None, model = None):
    if not filenames: filenames = load_haarlem_top_images()
    if not processor or not model: 
        processor, model = viti.load_processor_and_model()
    for filename in progressbar(filenames):
        output_filename = locations.haarlem_viti_outputs 
        output_filename += filename.split('/')[-1].replace('.jpg','.json')
        if os.path.isfile(output_filename): continue
        image = image_util.load_image(filename)
        output = viti.handle_image(image, processor, model)
        viti.save_outputs(output, output_filename)

def classify_all_pages(filenames = None):
    if not filenames: filenames = load_haarlem_viti_outputs()
    clf = viti.load_stamp_classifier()
    for filename in progressbar(filenames):
        pass

