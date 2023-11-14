from transformers import ViTImageProcessor, ViTForImageClassification
from image_util import to_color_image
import pickle


def load_processor(name = 'google/vit-base-patch16-224'):
    processor = ViTImageProcessor.from_pretrained(name)
    return processor

def load_model(name = 'google/vit-base-patch16-224'):
    model = ViTForImageClassification.from_pretrained(name)
    return model

def process_image(processor, image):
    if hasattr(image,'shape'): 
        if len(image.shape) == 2: image = to_color_image(image)
    inputs = processor(images = image, return_tensors="pt")
    return inputs

def predict(model, inputs, output_hidden_states = True):
    outputs = model(**inputs, output_hidden_states = output_hidden_states)
    return outputs

def handle_image(image, processor, model, output_hidden_states = True):
    inputs = process_image(processor, image)
    outputs = predict(model, inputs, output_hidden_states)
    return outputs

def handle_data(data, processor, model, output_hidden_states = True):
    top_image = data.top_image
    outputs = handle_image(top_image, processor, model, 
        output_hidden_states)
    return outputs

def save_outputs(outputs, filename):
    pickle.dump(outputs, open(filename, 'wb'))

def load_outputs(filename):
    with open(filename, 'rb') as f:
        outputs = pickle.load(f)
    return outputs
