from transformers import ViTConfig, ViTForImageClassification
from modules.utils import get_label2id_id2label

label2id, id2label = get_label2id_id2label()

config = ViTConfig(
    image_size=384,
    hidden_size=384,
    intermediate_size=1536,
    num_hidden_layers=12,
    num_attention_heads=6,
    patch_size=16,
    num_labels=len(label2id),
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    classifier_dropout_prob=0.0,
    initializer_range=0.02,
    qkv_bias=True,
    hidden_act="gelu",
)

model = ViTForImageClassification(
    config
)

model.config.id2label = id2label
model.config.label2id = label2id
model.config.problem_type = "multi_label_classification"

import torch.nn as nn

# model.classifier = nn.Sequential(
#     nn.Dropout(p=0.4, inplace=True),
#     nn.Linear(in_features=model.config.hidden_size, out_features=model.config.num_labels, bias=True),
# )


# print parameter count 
print('Number of parameters:', model.num_parameters())

print(model)

# save model to disk
model.save_pretrained('./models/T12')