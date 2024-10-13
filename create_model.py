from transformers import ViTConfig, ViTForImageClassification
from modules.utils import get_label2id_id2label

label2id, id2label = get_label2id_id2label()

# config = ViTConfig(
#     image_size=448,
#     hidden_size=768,
#     intermediate_size=3072,
#     num_hidden_layers=12,
#     num_attention_heads=12,
#     patch_size=16,
#     num_labels=len(label2id),
#     layer_norm_eps=1e-12,
#     hidden_dropout_prob=0.0,
#     attention_probs_dropout_prob=0.0,
#     classifier_dropout_prob=0.0,
#     initializer_range=0.02,
#     hidden_act="gelu",
# )

model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=len(label2id),
    image_size=448,
    ignore_mismatched_sizes=True,
)

model.config.id2label = id2label
model.config.label2id = label2id
model.config.problem_type = "multi_label_classification"


# print parameter count 
print('Number of parameters:', model.num_parameters())

print(model)

# save model to disk
model.save_pretrained('./models/T12')