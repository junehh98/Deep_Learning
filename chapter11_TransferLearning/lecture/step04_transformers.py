# -*- coding: utf-8 -*-
"""
  <transformers 설치 과정> 
1. conda activate tensorflow # 가상환경 활성화 
2. pip install transformers # transformers 설치 
"""

from transformers import pipeline # data input > token -> model 
'''
 관련 사이트 : https://huggingface.co/t5-base
 학습된 모델을 이용하므로 메모리를 많이 차지함(Colab 사용 권장)  
'''

### 1. 감성분석/문서분류 : 비지도학습 
'''
감성분석 : 주어진 문장(문서)의 성질/성향을 분석(label : 긍정,중립,부정)
문서분류 : 주어진 문장(문서)를 적절하게 분류  
               -> 주어진 dataset에 따라 category와 topic 달라짐
'''
# 1) 감성분석 
sentiment = pipeline(task = "sentiment-analysis")

pred = sentiment("what a beautiful day!") 
pred = pred[0] # dict 반환 

print(f'감성분석 결과 : {pred["label"]}, 감성 점수 : {pred["score"]}')

# 2) 문서분류 
classifier = pipeline(task = "text-classification") 
texts = ["This restaurant is awesome", "This restaurant is awful"]

preds = classifier(texts) # args (`str` or `List[str]`)
preds

for pred, text in zip(preds, texts) :
    print(f'{text} -> 분류결과 : {pred["label"]}, 감성 점수 : {pred["score"]}')



### 2. 텍스트 문서생성 
text_generator = pipeline(task = "text-generation") # or model="gpt2"
   
text_inputs =  """The COVID-19 pandemic, The novel virus was first identified in an outbreak in the Chinese city of Wuhan in 
December 2019."""

pred_text = text_generator(text_inputs)
print(pred_text) # list of `dict`

pred_text = pred_text[0]['generated_text'] # value 반환 
generated_text = pred_text.removeprefix(text_inputs) # input text 제거 

'''
import sys 
import time
for i in range(len(generated_text)):      
    next_char = generated_text[i] # 다음 글자 예측 
    time.sleep(0.1) # 0.1초 interval
    sys.stdout.write(next_char) # 줄바꿈없이 바로 이어서 출력
    sys.stdout.flush()
'''    
    
### 3. 질의응답(question-answering) 
question_answerer = pipeline(task = "question-answering")
'''
추출형 질의응답(extractive question answering)
문서에 대해서 질문을 제시하고 문서 자체에 존재하는 텍스트 범위(spans of text)를 
해당 질문에 대한 답변으로 추출하는 작업
'''
   
context = """Text mining, also referred to as text data mining, similar to text analytics,
is the process of deriving high-quality information from text. It involves
the discovery by computer of new, previously unknown information,
by automatically extracting information form different written resources."""


question = input('input question : ')
answer = question_answerer(question=question, context=context)
print(answer['answer'])


### 4. 문서 요약(summarization)
summarizer = pipeline(task = "summarization")

texts ="""Deep Neural Networks, also called convolutional networks, are composed of multiple levels of nonlinear operations, such as neural nets with many hidden layers. Deep learning methods aim at learning feature hierarchies, where features at higher levels of the hierarchy are formed using the features at lower levels. In 2006, Hinton et al. proved that much better results could be achieved in deeper architectures when each layer is pretrained with an unsupervised learning algorithm. Then the network is trained in a supervised mode using back-propagation algorithm to adjust weights. Current studies show that DNNs outperforms GMM and HMM on a variety of speech processing tasks by a large margin"""

result_texts = summarizer(texts)  # args (`str` or `List[str]`):           

summary_text = result_texts[0]['summary_text']
print(summary_text)

    
### 5. 문서번역(translation) 
'''
 1) 영어 -> 프랑스어 번역 
'''
translator_fr = pipeline(task = "translation_en_to_fr")
        
result_texts = translator_fr("How old are you?")
translation_text = result_texts[0]['translation_text']
print(translation_text)

'''
 2) 영어 -> 독일어 번역 
'''
translator_de = pipeline(task = "translation_en_to_de")

result_texts = translator_de("How old are you?")
translation_text = result_texts[0]['translation_text']
print(translation_text)

# 긴 문장 텍스트 
long_text = "The process of handling text data is a little different compared to other problems. This is because the data is usually in text form. You ,therefore, have to figure out how to represent the data in a numeric form that can be understood by a machine learning model. In this article, let's take a look at how you can do that. Finally, you will build a deep learning model using TensorFlow to classify the text."
translator_de(long_text)


### 6. 개체명 인식(Named-entity recognition)
'''
 - 문장에서 여러 개체(entity)를 감지하여 인식된 각 엔터티의 끝점과 신뢰도 점수 반환
 - 엔티티(entity) : 사람의 이름, 조직, 위치
'''
ner = pipeline("ner")

text = "John works for Google that is located in the USA"
result_texts = ner(text)
result_texts
