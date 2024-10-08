import json
import pandas as pd

def get_label2id_id2label(path='data/labels.csv'):
    # load labels
    labels = pd.read_csv(path)
    
    # create a label to id mapping
    label_to_id = {label: idx for idx, label in enumerate(labels['name'])}
    
    # create an id to label mapping
    id_to_label = {idx: label for idx, label in enumerate(labels['name'])}
    
    return label_to_id, id_to_label