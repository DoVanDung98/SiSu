import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import re
from utils.constant import df

print(df.head())
df = shuffle(df)
print(df.head(20))

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r"[^a-zA-z.!?'0-9]", ' ', text)
    text = re.sub('\t', ' ',  text)
    text = re.sub(r" +", ' ', text)
    return text

df['review'] = df['review'].apply(clean_text)
rcParams['figure.figsize'] = 8, 6
sns.countplot(df.sentiment)
plt.xlabel('review score');

df['text'] = df['text'].apply(clean_text)
rcParams['figure.figsize'] = 8, 6
sns.countplot(df.category_2)
plt.xlabel('review score');

rcParams['figure.figsize'] = 8, 6
sns.countplot(df.category_1)
plt.xlabel('review score');