# -*- coding: utf-8 -*-
"""
문) 네이버 url(https://naver.com)을 대상으로 검색 입력 상자에 임의의 셀럼 이름을
 입력하여 이미지 10장을 셀럼 이름의 폴더에 저장하시오.  
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys # 엔터키
from selenium.webdriver.common.by import By 
from urllib.request import urlretrieve # image save
import os # 폴더 생성 및 이동 

def naver_img_crawler(name) :
    # 1. driver 객체 생성 
    path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32"
    driver = webdriver.Chrome(path + "/chromedriver.exe")
    
    # 2. 대상 url 
    driver.get("https://naver.com") # naver url 이동 
    
    # 3. 검색상자 : name 속성으로 element 가져오기 
    query_input = driver.find_element(By.NAME, "query") # 입력상자  
    
    # 4. 검색어 입력 -> 엔터 
    query_input.send_keys(name) # 검색어 입력 
    query_input.send_keys(Keys.ENTER) # 통합 검색결과 페이지 이동     
    
    # 5. 이미지 링크 가져오기 
    # <a role="tab" href="?where=image&amp;sm=tab_jum&amp;query=%EB%B0%95%EB%82%98%EB%9E%98" onclick="return goOtherCR(this,'a=tab*i.jmp&amp;r=2&amp;i=&amp;u='+urlencode(this.href));" class="tab" aria-selected="false">이미지</a>      
    image_link = driver.find_element(By.LINK_TEXT, '이미지') 
    image_link.click()
    #image_link.send_keys(Keys.ENTER) # 링크 클릭 
    driver.implicitly_wait(3) # 3초 대기(자원 loading)
    
    # 6. 이미지 태그와 url 수집 
    '''
    div/img 태그 xpath 
    //*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[1]/div/div[1]/a/img
    //*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[2]/div/div[1]/a/img
    '''
    image_urls = []
    for i in range(15) : # 예외발생 고려 
        try :
            # img 태그 수집 
            img_tag = driver.find_element(By.XPATH, f'//*[@id="main_pack"]/section[2]/div/div[1]/div[1]/div[{i}]/div/div[1]/a/img')
            # src 속성 수집 
            img_url = img_tag.get_attribute('src')
            # image url 저장 
            image_urls.append(img_url) 
        except :
            print('예외발생')
    
    print('url 개수 :', len(image_urls))    
    
    # 7. image 저장 폴더 생성과 이동 
    pwd = r'C:\ITWILL\6_DeepLearning\workspace\chapter07_Selenium_ImageCrawling' # 저장 경로 
    os.mkdir(pwd + '/' + name) # pwd 위치에 폴더 생성(셀럽이름) 
    os.chdir(pwd+"/"+name) # 폴더 이동(현재경로/셀럽이름)
        
    # 8. image url -> image save
    for i in range(len(image_urls)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg" # test1.jsp
            # server image -> file save
            urlretrieve(image_urls[i], filename=file_name)#(url, filepath)
            print(str(i+1) + '번째 image 저장')
        except :
            print('해당 url에 image 없음 : ', image_urls[i])        


    driver.close() # driver 닫기 



# 함수 호출 
name_list = ['강호동', '박나래', '사자']
for name in name_list :
    naver_img_crawler(name)   
