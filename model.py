from transformers import XLNetForSequenceClassification
from utils.device import device


model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels = 2)
model = model.to(device)

