# -*- coding: utf-8 -*-
"""
step02_button_click.py

1. naver page 이동 
2. login 버튼 클릭 
3. 페이지 이동(뒤로, 앞으로) 
"""

from selenium import webdriver # driver  
from selenium.webdriver.common.by import By # 로케이터 
import time # 화면 일시 정지 

# 1. driver 객체 생성 
path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32" # driver 경로 
driver = webdriver.Chrome(path + '/chromedriver.exe')


# 2. 대상 url 이동 
driver.get('https://www.naver.com/') # url 이동 



# 3. 로그인 버튼 태그(element) 가져오기 : xpath로 가져오기(ppt.14)  
'''  
 상대경로 메뉴 : 복사(copy) -> xpath 복사(copy xpath) 
 절대경로 메뉴 : 복사(copy) -> 전체 xpath 복사(copy full xpath)
'''

login_btn = driver.find_element(By.XPATH, '//*[@id="account"]/div/a') 

login_btn.click() # 버튼 클릭 
time.sleep(2) # 2초 일시 중지

driver.back() # 현재페이지 -> 이전으로
time.sleep(2) # 2초 일시 중지 
  
driver.forward() # 이전 -> 앞으로 
driver.refresh() # 페이지 새로고침(F5)

time.sleep(2) # 2초 일시 중지 
driver.close() # 현재 창 닫기  


