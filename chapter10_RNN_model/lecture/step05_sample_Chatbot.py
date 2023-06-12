# -*- coding: utf-8 -*-
"""
step05_sample_Chatbot.py

간단한 챗봇 만들기 : PPT.34 참고 
"""

# 작업순서 
# texts -> 토큰 -> 정수색인 -> padding -> model -> y_pred  

from tensorflow.keras.preprocessing.text import Tokenizer # 토큰 생성기 
from tensorflow.keras.preprocessing.sequence import pad_sequences # 패딩 
from tensorflow.keras.utils import to_categorical # y변수 : one-hot 인코딩 

# RNN + DNN model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM 

# PPT 참고 
text="""이름은 홍길동
취미는 음악감상 독서 등산
직업은 강사 프로그래머"""

# 1. 토큰(token)
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text]) # 텍스트 반영 
words = tokenizer.word_index # 토큰(단어)
print(words)
'''
{'이름은': 1, '홍길동': 2, '취미는': 3, '음악감상': 4, '독서': 5, '등산': 6, '직업은': 7, '강사': 8, '프로그래머': 9}
'''

# 전체 단어수 + 1
words_size =  len(words) + 1
print('전체 단어수 =', words_size) # 전체 단어수 = 10


# 2. 정수색인 -> 순차 data(RNN 공급)
'''
이름은 -> 홍길동 
취미는 -> 음악감상
취미는 음악감상 -> 독서
취미는 음악감상 독서 -> 등산
직업은 -> 강사 -> 프로그래머
직업은 강사 -> 프로그래머
'''

sequences = []
for line in text.split('\n') : # 이름은 홍길동
    # 1) 정수 인덱스 
    seq = tokenizer.texts_to_sequences([line])[0]
 
    # 2) sequence data 생성
    for i in range(1, len(seq)) : # 1, 3, 2
        sequence = seq[:i+1] # [0~1], [0~1, 0~2, 0~3], [0~1, 0~2] 
        sequences.append(sequence)
        
print(sequences) # 학습할 문장 = 6개        
   
    
# 3. padding(인코딩) : maxlen -> 0 padding
lens = [len(s) for s in sequences]
maxlen = max(lens)
print(maxlen) # 4

# 최대 단어 길이 이용 padding 
sequences = pad_sequences(sequences, maxlen = maxlen)
print(sequences)


# 4. X, y변수 생성 
X = sequences[:,:-1]
X.shape # (6, 3) -> (1, 6, 3) : (batch_size, time_steps, features)

y = sequences[:,-1]

# 10진수 -> one-hot encoding 
y = to_categorical(y, num_classes=words_size)


# 5. model 생성 
model = Sequential()


# 6. embedding : 정수 인덱스 -> 인코딩 
model.add(Embedding(input_dim=words_size, output_dim=4, 
                    input_length=maxlen-1))
'''
input_dim : 전체 단어수 
output_dim : embedding vector 크기(차원)
input_length : 1문장을 구성하는 단어 길이(maxlen-1 : y변수 빼줌) 
'''

# 7. RNN(순환신경망)
model.add(LSTM(units = 32, activation='tanh'))


# 8. output layer 
model.add(Dense(units=words_size, activation='softmax'))


# 9. model 학습환경
model.compile(optimizer='adam',
              loss='categorical_crossentropy', # y : one hot encoding 
              metrics=['accuracy'])

# 10. model 학습 
model.fit(X, y, epochs=200, verbose=1)


# 문장 생성기 : # many-to-one.txt 참고 
def sentence_generation(search_word, n=0): # 검색 단어, 단어길이
    global model, tokenizer, maxlen, X
      
    seq_idx = tokenizer.texts_to_sequences([search_word])[0]      
    # 검색 단어 길이 결정  
    for row in X :
        if seq_idx in row : 
                n += 1
                
    if n == 0 : # "해당 단어 없음"
        return 0
    
    sentence = '' # 문장 save
    start_word = search_word # 검색 단어 변수 저장 

    for i in range(n): # n번 반복      
        # 1 인코더 : 검색단어 -> 정수 인덱스 
        seq_idx = tokenizer.texts_to_sequences([search_word])[0]#현재 단어 정수인코딩
        encoded = pad_sequences([seq_idx], maxlen=maxlen-1) # 데이터에 대한 패딩
        
        # 2. RNN model 예측 
        result = model.predict_classes(encoded, verbose=0)   

        # 3. 디코더 : 예측치(인코딩) -> 단어 변환 
        for word, index in tokenizer.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        search_word = search_word + ' '  + word # 현재 단어+ ' '+예측 단어 ->현재 단어 변경
        sentence = sentence + ' ' + word # 예측 단어 문장 생성
    
    sentence = start_word + sentence
    return sentence



# 검색 단어 입력 
while(True) :
    result = sentence_generation(input("검색단어 : "))#이름은, 취미는, 직업은 
    print(result)
    if result == 0 :
        print("해당 단어 없음")
        break







