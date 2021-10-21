from transformers import XLNetTokenizer, XLNetModel
from utils.constant import MAX_LEN
from keras.preprocessing.sequence import pad_sequences
import torch 
from utils.device import device
from model import model
import torch.nn.functional as F 
class_names = ['Relationship','Spiritual']

PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

def predict_sentiment(text):
    review_text = text

    encoded_review = tokenizer.encode_plus(
    review_text,
    max_length=MAX_LEN,
    add_special_tokens=True,
    return_token_type_ids=False,
    pad_to_max_length=False,
    return_attention_mask=True,
    return_tensors='pt',
    )

    input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    input_ids = input_ids.astype(dtype = 'int64')
    input_ids = torch.tensor(input_ids) 

    attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
    attention_mask = attention_mask.astype(dtype = 'int64')
    attention_mask = torch.tensor(attention_mask) 

    input_ids = input_ids.reshape(1,512).to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    outputs = outputs[0][0].cpu().detach()

    probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
    _, prediction = torch.max(outputs, dim =-1)

    print("Spiritual score:", probs[1])
    print("Relationship score:", probs[0])
    print(f'SiSu text: {review_text}')
    print(f'Sentiment  : {class_names[prediction]}')

# Input text Sisu sentiment
text = "Every time I need to fill a paper for the government for applying something, or doing a report or something, that takes a little bit of sisu. And also like this kind of this kind of situation that if something goes totally south, so like, there's a huge mistake or something and you're like,you're focused on something else, and you need to shift to something that you know, that this needs to be solved, but you just don't want to. You're just not in a mood but you k now, that this has to be done. So you just think Okay, let's do this"
predict_sentiment(text)