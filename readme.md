# SiSu competition  

- SiSu competition sentiment classification used data clone from youtube, tedtalk...  
- Used XLNet model pretrained  
- Used 2046 lines text converted csv file  
- Split data from ratio (0.7, 0.15, 0.15) -> (train, val, test) consists of two categorical:  
`categorical1: good sisu and bad sisu`  
`categorical1: Spiritual and Relationship`  

## Implement project sisu sentiment classification  
`step1: git clone https://github.com/DoVanDung98/SiSu.git`  
`step2: pip install -r requirements.txt`  
`step3: download weight from my drive.  
Link drive: https://drive.google.com/drive/u/0/folders/1WrniGZcVJDe0WjT02NadfyFUVLBjh2Bf`
`step4: python train.py`  
`step5: python test.py`  

I'll design jupyter notebook file. Can you use SiSu.ipynb for your target(train, val, test) model.  

## For example test SiSu:  
`text = "This is the best movie I have ever seen!! The story is such a motivation"`  

`predict_sentiment(text)`

## Result:  

`Spiritual score: 0.9981069564819336  
Relationship score: 0.001893064472824335  
SiSu text: This is the best movie I have ever  seen!! The story is such a motivation
Sentiment  : Relationship`

#### Created by: VanDung and MinhThy