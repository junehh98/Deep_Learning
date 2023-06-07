# -*- coding: utf-8 -*-
"""
step04_xpath_attribute.py

<작업순서> 
 google -> 입력상자 -> 검색어 입력 -> [검색 페이지 이동] -> element 수집 
"""

from selenium import webdriver # driver 생성  
from selenium.webdriver.common.by import By # By.NAME,  By.TAG_NAME
from selenium.webdriver.common.keys import Keys # 엔터키 역할 

'''
a 태그 url xpath
//*[@id="rso"]/div[1]/div/div/div[1]/div/a -> 1번째 링크 
//*[@id="rso"]/div[2]/div/div/div[1]/div/a -> 2번째 링크
   :
//*[@id="rso"]/div[8]/div/div/div/div/div/div[1]/div[1]/div/a  : 예외발생      
//*[@id="rso"]/div[10]/div/div/div[1]/div/a -> 10번째 링크       

i = 1 ~ 10
f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/a' 

a 내용 xpath 
//*[@id="rso"]/div[1]/div/div/div[1]/div/a/h3
//*[@id="rso"]/div[2]/div/div/div[1]/div/a/h3

i = 1 ~ 10
f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/a/h3' 
'''

# 키워드 관련 url 수집 함수 
def keyword_search(keyword) :
    # 1. driver 객체 생성 
    path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32" 
    driver = webdriver.Chrome(path + '/chromedriver.exe')
    
    # 2. 대상 url 이동 
    driver.get('https://www.google.com/') # url 이동   
    
    # 3. 검색어 입력상자 : name 속성으로 가져오기     
    input_text = driver.find_element(By.NAME, 'q') # 1개 element 
    
    # 4. 검색어 입력 -> 엔터 
    input_text.send_keys(keyword)
    input_text.send_keys(Keys.ENTER) # 엔터키 누름 -> 검색 페이지 이동 
    driver.implicitly_wait(3) # 3초 대기(자원 loading)
    
    # 5. 검색 페이지 element 수집 : tag 이름으로 가져오기 
    a_urls = []  # url  
    a_conts = [] # 내용             
    for i in range(1, 11) :#XPATH가 다른 경우 : 예외처리 
        try : 
            a_tag = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/a')
            h3_tag = driver.find_element(By.XPATH, f'//*[@id="rso"]/div[{i}]/div/div/div[1]/div/a/h3')
            a_urls.append(a_tag.get_attribute('href')) # url
            a_conts.append(h3_tag.text) # tag 내용 
            
        except :
            print('예외발생')

    print('수집 urls 개수 =', len(a_urls)) # 수집 elements 개수 = 6
    driver.close() # 창 닫기 
    
    return a_urls, a_conts


# 함수 호출 
keyword = input('검색어 입력 : ') # 파이썬, 크롤링  
a_urls, a_conts = keyword_search(keyword)
print('-'*40)
print(a_urls)
print(a_conts)













