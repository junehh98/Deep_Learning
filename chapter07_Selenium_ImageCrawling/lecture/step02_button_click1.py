# -*- coding: utf-8 -*-
"""
step02_button_click1.py

로케이터(locator) : By 클래스에서 제공되는 요소(element) 찾기 기능 

<작업순서>
1. naver page 이동 
2. login 버튼 클릭 
3. 페이지 이동(뒤로, 앞으로) 
"""

from selenium import webdriver # driver  
from selenium.webdriver.common.by import By # 로케이터(locator)
import time # 화면 일시 정지 

# 1. driver 객체 생성 
path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32" # driver 경로 
driver = webdriver.Chrome(path + '/chromedriver.exe')

# 2. 대상 url 이동 
driver.get('https://www.naver.com/') # url 이동 
dir(driver)
'''
find_element(By.로케이터, '이름') : 1개 태그(element) 수집 
find_elements(By.로케이터, '이름') : 모든 태그(element) 수집 
'''

# 3. 로그인 버튼 태그(element) 가져오기 : class name으로 가져오기(ppt.13) 
login_btn = driver.find_element(By.CLASS_NAME, "MyView-module__link_login___HpHMW")
# <a href="https://nid.naver.com/nidlogin.login?mode=form&amp;url=https://www.naver.com/" class="MyView-module__link_login___HpHMW"><i class="MyView-module__naver_logo____Y442"><span class="blind">NAVER</span></i>로그인</a>
login_btn.click() # 로그인 버튼 클릭 
time.sleep(2) # 2초 일시 중지 

driver.back() # 현재페이지 -> 뒤로
time.sleep(2) # 2초 일시 중지 
  
driver.forward() # 현재페이지 -> 앞으로 
driver.refresh() # 페이지 새로고침(F5)
time.sleep(2) # 2초 일시 중지 

driver.close() # 현재 창 닫기  


