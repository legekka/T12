from transformers import EfficientNetForImageClassification, EfficientNetConfig
from modules.utils import get_label2id_id2label

from torch.nn import Linear
import json

label2id, id2label = get_label2id_id2label()

config = EfficientNetConfig(
    id2label=id2label,
    image_size=448,
)

model = EfficientNetForImageClassification(config)


model.config.id2label = id2label
model.config.label2id = label2id
model.config.problem_type = "multi_label_classification"
model.config.num_classes = len(label2id)

# model.classifier = Linear(
#     in_features=model.classifier.in_features,
#     out_features=len(label2id),
#     bias=True
# )

# print parameter count 
print('Number of parameters:', model.num_parameters())
print('Number of classes:', model.config.num_classes)

# print(model)

# save model to disk
model.save_pretrained('./models/T12-EfficientNet')

preprocessor_config = {
    "do_normalize": True,
    "do_resize": True,
    "image_mean": [0.673, 0.608, 0.602],
    "image_std": [0.253, 0.250, 0.239],
    "size": 448,
}

with open('./models/T12-EfficientNet/preprocessor_config.json', 'w') as f:
    json.dump(preprocessor_config, f)

# unload model
model = None

# load model from disk
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained('./models/T12-EfficientNet')

print('Model loaded successfully!')