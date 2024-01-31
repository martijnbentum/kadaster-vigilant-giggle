import glob
import image_util
import locations
import numpy as np
import os
import perceptron
from progressbar import progressbar
import tesseract
import viti

def load_haarlem_files():
    '''returns a list of filenames of input images'''
    fn = glob.glob(locations.haarlem + '**/*.jpg', recursive=True)
    return fn

def load_haarlem_pages():
    '''returns a list of filenames of the pages'''
    fn = glob.glob(locations.haarlem_split_pages + '*.jpg')
    return fn

def load_haarlem_top_images():
    '''returns a list of filenames of the top images of the pages'''
    fn = glob.glob(locations.haarlem_top_images + '*.jpg')
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

def get_viti_output_filenames():
    '''returns a list of filenames of the viti model output'''
    return glob.glob(locations.haarlem_viti_outputs + '*.json')

def batch_viti_filenames(filenames = None, batch_size = 1000):
    '''returns a list of lists of filenames, where each list has length
    batch_size'''
    if not filenames: filenames = get_viti_output_filenames()
    batched_filenames = np.array_split(filenames, 
        np.ceil(len(filenames)/ batch_size))
    return batched_filenames

def load_viti_data(fn):
    '''loads the logits of the viti model output for the given filenames'''
    X = np.zeros([len(fn), 1000])
    for index, filename in enumerate(fn):
        X[index] = perceptron._get_layer(filename)
    return X

def classify(fn, clf):
    '''classifies viti data with the stamp classifier
    the classifier can be loaded with perceptron.load_classifier()
    '''
    # strip filename of directory and extension
    names = [f.split('/')[-1].split('.')[0] for f in fn]
    # get the viti data
    X = load_viti_data(fn)
    hyp = clf.predict(X)
    # link the names to the predictions
    output = list(zip(list(map(str,hyp)), names))
    return output


def add_filenames_to_log(filenames):
    '''adds filenames to the log file'''
    with open(locations.haarlem_base + 'viti_done.log', 'a') as f:
        for filename in filenames:
            f.write(filename + '\n')

def load_viti_done_log():
    if not os.path.exists(locations.haarlem_base + 'viti_done.log'):
        return []
    with open(locations.haarlem_base + 'viti_done.log', 'r') as f:
        log = f.read().split('\n')
    return log

def check_filename_in_log(filename, log = None):
    '''checks if the filename is in the log file'''
    if not log: log = load_viti_done_log()
    return filename in log

def exclude_done_filenames(filenames):
    '''returns a list of filenames that are not in the log file'''
    log = load_viti_done_log()
    return [f for f in filenames if not check_filename_in_log(f, log)]

def write_classification_output(output):
    '''writes the classification output to a log file
    the output is generated by the classify function
    '''
    with open(locations.haarlem_base + 'classification.log', 'a') as f:
        for line in output:
            f.write('\t'.join(line) + '\n')

def load_classifications():
    '''loads the classification log file
    in the file is the name of the page and the classification
    the filename can be found by adding the extension .jpg
    and prepending the directory locations.haarlem_split_pages
    '''
    with open(locations.haarlem_base + 'classification.log', 'r') as f:
        classifications= f.read().split('\n')
    return classifications

def classify_all_pages(filenames = None):
    '''classifies all pages with the stamp classifier
    the classification results are written to a classification log file
    it skips files that are already classified (i.e. in the viti done log file)
    '''
    if not filenames: filenames = get_viti_output_filenames()
    print('total n files:', len(filenames))
    filenames = exclude_done_filenames(filenames)
    print('n files to do:', len(filenames))
    batched_filenames = batch_viti_filenames(filenames)
    clf = perceptron.load_classifier()
    for batch in progressbar(batched_filenames):
        output = classify(batch, clf)
        write_classification_output(output)
        add_filenames_to_log(batch)

