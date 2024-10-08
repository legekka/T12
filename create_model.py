from transformers import ViTConfig, ViTForImageClassification
from modules.utils import get_label2id_id2label

label2id, id2label = get_label2id_id2label()

config = ViTConfig(
    image_size=512,
    hidden_size=768,
    intermediate_size=3072,
    num_hidden_layers=12,
    num_attention_heads=12,
    patch_size=24,
    num_labels=len(label2id),
    layer_norm_eps=1e-12,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    classifier_dropout_prob=0.1,
    hidden_act="gelu",
)
model = ViTForImageClassification(
    config,
)

model.config.id2label = id2label
model.config.label2id = label2id
model.config.problem_type = "multi_label_classification"

# print parameter count 
print('Number of parameters:', model.num_parameters())

# save model to disk
model.save_pretrained('./models/T12')