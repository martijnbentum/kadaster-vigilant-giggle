import json
import locations
import os
from progressbar import progressbar
import shutil

def copy_image_to_classification_folder(top_image = False):
    '''copy the images to the correct classification folder'''
    classifications = load_log_classification_from_json()
    names, doubles = [], []
    for line in progressbar(classifications):
        name = line['name']
        if name in names: doubles.append(name)
        else: names.append(name)
        if top_image:
            input_dir = locations.haarlem_top_images
        else:
            input_dir = locations.haarlem_split_pages
        input_filename = input_dir + name + '.jpg'
        if line['stamped']:
            if top_image:
                output_dir = locations.haarlem_top_with_stamp
            else:
                output_dir = locations.haarlem_with_stamp
        else:
            if top_image:
                output_dir = locations.haarlem_top_no_stamp
            else:
                output_dir = locations.haarlem_no_stamp
        output_filename = output_dir + name + '.jpg'
        shutil.copy(input_filename, output_filename)

def load_log(filename = locations.fn_log):
    '''load the tesseract log file
    that indicates how well the image was processed
    '''
    with open(filename) as f:
        temp = f.read().split('\n')
    output = [log_line_to_dict(x) for x in temp if x]
    return output 

def load_classifications(filename = locations.fn_classification):
    '''load the stamp classifications file
    contains a classification of the image with the name of the image
    '''
    with open(filename) as f:
        temp = f.read().split('\n')
    output = []
    for line in temp:
        if line:
            output_line = {}
            line = line.split('\t')
            output_line['stamped'] = bool(int(line[0]))
            output_line['name'] = line[1]
            output.append(output_line)
    return output

def filename_to_name(filename):
    '''maps a filename to a name.'''
    return filename.split('/')[-1].split('.')[0]

def str_to_bool(string):
    '''converts a string True or False to a boolean'''
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError('String is not True or False', string)

def log_line_to_dict(line):
    if type(line) == str: line = line.split('\t')
    output = {}
    output['name'] = filename_to_name(line[0])
    output['top_rectangle_present'] = str_to_bool(line[2])
    output['name_rectangle_present'] = str_to_bool(line[3])
    output['special_dict'] = line[4].split(',')
    output['top_dict'] = line[5].split(',')
    return output

def link_log_classification(log, classification):
    output = []
    for classification_line in classification:
        output_line = {}
        for log_line in log:
            name = log_line['name']
            if name == classification_line['name']:
                output_line = log_line.copy()
                output_line['stamped'] = classification_line['stamped']
                output_line['log_available'] = True
        if not output_line:
            output_line = classification_line.copy()
            output_line['log_available'] = False
        output.append(output_line)
    return output

def save_log_classification_to_json(log_classification = None):
    if not log_classification: log_classification  = link_log_classification(
        load_log(), load_classifications())
    with open(locations.fn_log_classification, 'w') as f:
        json.dump(log_classification, f)

def load_log_classification_from_json():
    with open(locations.fn_log_classification) as f:
        return json.load(f)

def name_to_top_image(name):
    return locations.haarlem_top_images + name + '.jpg'
