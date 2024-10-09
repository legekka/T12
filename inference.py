from transformers import ViTForImageClassification, AutoImageProcessor
import argparse
import torch
import torchvision.transforms as T

from PIL import Image

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('-m','--model_path', type=str, default='./models/T12', help='Path to the model')
parser.add_argument('-i','--image_path', type=str, default='./data/test.jpg', help='Path to the image')

args = parser.parse_args()

image_processor = AutoImageProcessor.from_pretrained(args.model_path, use_fast=True)
model = ViTForImageClassification.from_pretrained(args.model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_val_transforms = T.Compose([
    T.Resize(size), 
    T.ToTensor(), 
    T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])

image = Image.open(args.image_path)
image = _val_transforms(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image)

    # sigmoid to get probabilities
    probs = torch.sigmoid(outputs.logits)

    # get the predicted labels
    predicted_labels = probs > 0.5
    predicted_labels = predicted_labels.cpu().numpy()

    # get the predicted labels with probabilities
    predicted_labels = [(model.config.id2label[i],
                            probs[0][i].item())
                          for i in range(len(predicted_labels[0])) if predicted_labels[0][i]]


    # sort the labels by probability
    labels_with_probs = sorted(predicted_labels, key=lambda x: x[1], reverse=True)

    for label, prob in labels_with_probs:
        print(f"{label}: {prob:.4f}")
