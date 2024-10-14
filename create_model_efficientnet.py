from transformers import EfficientNetForImageClassification, EfficientNetConfig
from modules.utils import get_label2id_id2label

from torch.nn import Linear

label2id, id2label = get_label2id_id2label()

config = EfficientNetConfig(
    depth_coefficient=1.8,
    depthwise_padding=[6],
    dropout_rate=0.4,
    hidden_act="swish",
    hidden_dim=1792,
    id2label=id2label,
    image_size=384,
    label2id=label2id,
    width_coefficient=1.4
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

print(model)

# save model to disk
model.save_pretrained('./models/T12-EfficientNet')

# unload model
model = None

# load model from disk
from transformers import AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained('./models/T12-EfficientNet')

print('Model loaded successfully!')