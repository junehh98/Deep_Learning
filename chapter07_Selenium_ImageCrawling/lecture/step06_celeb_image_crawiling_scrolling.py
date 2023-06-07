# -*- coding: utf-8 -*-
"""
step06_celeb_image_crawiling_scrolling.py

셀럽 이미지 수집 
 Selenium + Driver + BeautifulSoup + Screen scroll
"""

from selenium import webdriver
from selenium.webdriver.common.by import By # By.NAME,  By.TAG_NAME

from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
from bs4 import BeautifulSoup
from urllib.request import urlretrieve # server image 
import os # dir 경로/생성/이동
import time

def celeb_crawler(name) :    
    # 1. dirver 경로 지정 & 객체 생성  
    path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32"
    driver = webdriver.Chrome(path + "/chromedriver.exe")
    
    # 1. 이미지 검색 url 
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
    
    # 2. 검색 입력상자 tag -> 검색어 입력   
    search_box = driver.find_element(By.NAME, 'q')
    search_box.send_keys(name) # 검색어 입력  
    search_box.send_keys(Keys.ENTER) # 검색창에서 엔터 
    
    driver.implicitly_wait(3) # 3초 대기(자원 loading)

    # ------------ 스크롤바 내림 ------------------------------------------------------ 
    last_height = driver.execute_script("return document.body.scrollHeight") #현재 스크롤 높이 계산
    
    while True: # 무한반복
        # 브라우저 끝까지 스크롤바 내리기
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") 
        
        time.sleep(2) # 2초 대기 - 화면 스크롤 확인
    
        # 화면 갱신된 화면의 스크롤 높이 계산
        new_height = driver.execute_script("return document.body.scrollHeight")

        # 새로 계산한 스크롤 높이와 같으면 stop
        if new_height == last_height: 
            try: # [결과 더보기] : 없는 경우 - 예외처리             
                driver.find_element(By.CLASS_NAME, "mye4qd").click() # [결과 더보기] 버튼 클릭 
            except:
                break
        last_height = new_height # 새로 계산한 스크롤 높이로 대체 
    #-------------------------------------------------------------------------
    
    # 3. 이미지 div 태그 수집  
    image_url = []
    for i in range(50) : # image 개수 지정               
        src = driver.page_source # 현재페이지 source 수집 
        html = BeautifulSoup(src, "html.parser")
        div_img = html.select_one(f'div[data-ri="{i}"]') 
    
        # 4. img 태그 수집 & image url 추출
        img_tag = div_img.select_one('img[class="rg_i Q4LuWd"]')
        try :
            image_url.append(img_tag.attrs['src'])
            print(str(i+1) + '번째 image url 추출')
        except :
            print(str(i+1) + '번째 image url 없음')
      
            
    # 5. 중복 image url 삭제      
    print(len(image_url)) # 43      
    image_url = list(set(image_url)) # 중복 url  삭제 
    print(len(image_url)) # 43   
    
    # 6. image 저장 폴더 생성과 이동 
    pwd = r'C:\ITWILL\\6_DeepLearning\workspace\chapter07_Selenium_ImageCrawling' # 저장 경로 
    os.mkdir(pwd + '/' + name) # pwd 위치에 폴더 생성(셀럽이름) 
    os.chdir(pwd+"/"+name) # 폴더 이동(현재경로/셀럽이름)
        
    # 7. image url -> image save
    for i in range(len(image_url)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg"           
            urlretrieve(image_url[i], filename=file_name)#(url, filepath)
            print(str(i+1) + '번째 image 저장')
        except :
            print('해당 url에 image 없음 : ', image_url[i])        
    
    driver.close() # driver 닫기 
    
    
# 1차 테스트 함수 호출 
#celeb_crawler("하정우")   

# 여러명의 셀럽 이미지 수집 
name_list = ["조인성", "송강호", "전지현"] # 31장, 30장, 30장 
# 40장, 35장, 48장 
for name in name_list :
    celeb_crawler(name)
    
    
    
    
    
    
    
    
    
    

    