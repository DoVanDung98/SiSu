from utils.constant import df
from transformers import XLNetTokenizer, XLNetModel
import seaborn as sns
import matplotlib.pyplot as plt
def sentiment2label(category_1):
    if category_1 == "Spiritual":
        return 1
    else :
        return 0

df['category_2'] = df['category_2'].apply(sentiment2label)

print(df['category_2'].value_counts())
class_names = ['Relationship','Spiritual']
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer = XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

input_txt = "India is my country. All Indians are my brothers and sisters"
encodings = tokenizer.encode_plus(input_txt, add_special_tokens=True, max_length=16, return_tensors='pt', 
return_token_type_ids=False, return_attention_mask=True, pad_to_max_length=False)

print('input_ids : ',encodings['input_ids'])
print(tokenizer.convert_ids_to_tokens(encodings['input_ids'][0]))
print(type(encodings['attention_mask']))
attention_mask = pad_sequences(encodings['attention_mask'], maxlen=512, dtype=torch.Tensor ,truncating="post",padding="post")
attention_mask = attention_mask.astype(dtype = 'int64')
attention_mask = torch.tensor(attention_mask) 
print(attention_mask.flatten())
print(encodings['input_ids'])
token_lens = []

for txt in df['text']:
    tokens = tokenizer.encode(txt, max_length=512)
    token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([2, 10]);
plt.xlabel('Token count');