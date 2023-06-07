# -*- coding: utf-8 -*-
"""
step05_celeb_image_crawling.py

셀럽 이미지 수집 
 Selenium + web Driver + BeautifulSoup
"""

from selenium import webdriver # driver 
from selenium.webdriver.common.by import By # By.NAME,  By.TAG_NAME
from selenium.webdriver.common.keys import Keys # 엔터키 사용(Keys.ENTER) 
from bs4 import BeautifulSoup # find, find_all, select
from urllib.request import urlretrieve # server image -> image save 
import os # dir 경로/생성/이동

def celeb_crawler(name) :    
    # 1. dirver 경로 지정 & 객체 생성  
    path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32"
    driver = webdriver.Chrome(path + "/chromedriver.exe")
    
    # 2. 이미지 검색 url 
    driver.get("https://www.google.co.kr/imghp?hl=ko&ogbl")
    
    # 3. 검색 입력상자 tag -> 검색어 입력   
    search_box = driver.find_element(By.NAME, 'q') # name='q'
    search_box.send_keys(name) # 검색어 입력     
    search_box.send_keys(Keys.ENTER) # 엔터키 : 검색 페이지 이동  
    driver.implicitly_wait(3) # 3초 대기(자원 loading)
    
    print('url :', driver.current_url) # 현재 접속한 url 확인 
    '''
    //*[@id="islrg"]/div[1]/div[1]
    <div data-ved="2ahUKEwja-6LXuKv_AhUEsFYBHRtHBhgQMygAegUIARDaAQ" jsaction="cFWHmd:s370ud;TMn9y:cJhY7b;" data-hveid="CAEQ2gE" data-ictx="1" data-id="KwUi6YA4UZ3ZSM" jsname="N9Xkfe" 
    data-ri="0" class="isv-r PNCib MSM1fd BUooTd" jscontroller="H9MIue" jsmodel="lbVNPd Whqy4b" style="width: 145px; height: 222px;" data-tbnid="KwUi6YA4UZ3ZSM" data-ct="0" data-cb="12" data-cl="3" data-cr="6" data-tw="201" data-ow="640" data-oh="800" data-sc="1" data-os="-2"><h3 class="bytUYc">하정우 - 나무위키</h3><a class="wXeWr islib nfEiy" jsname="sTFXNd" tabindex="0" role="button" jsaction="J9iaEb;mousedown:npT2md; touchstart:npT2md;" data-nav="1" style="height: 180px;"><div class="bRMDJf islir" jsname="DeysSe" style="height: 182px;"><img src="data:image/jpeg;base64,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" data-deferred="1" class="rg_i Q4LuWd" jsname="Q4LuWd" width="145" height="182" alt="하정우 - 나무위키" data-iml="765.4000000953674" data-atf="true"></div><div class="c7cjWc mvjhOe"></div></a><a class="VFACy kGQAp sMi44c d0NI4c lNHeqe WGvvNb" data-ved="2ahUKEwja-6LXuKv_AhUEsFYBHRtHBhgQr4kDegUIARDbAQ" jsname="uy6ald" rel="noopener" target="_blank" href="https://namu.wiki/w/%ED%95%98%EC%A0%95%EC%9A%B0" jsaction="focus:trigger.HTIQtd;mousedown:trigger.HTIQtd;touchstart:trigger.HTIQtd;" title="하정우 - 나무위키"><div class="fxgdke Yx2mie cS4Vcb-pGL6qe-k1Ncfe"><img src="data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBwgHBgkIBwEKCgkLDRYPDQEMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3Ojo6Iys/RD84QzQ5OjcBCgoKDQwNGg8PGjclHyU3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3N//AABEIABAAEAMBEQACEQEDEQH/xAAWAAADAAAAAAAAAAAAAAAAAAACAwT/xAAmEAABAgMIAgMAAAAAAAAAAAABAhEDBCEABQYHEhMUMVFxIkGx/8QAGAEBAAMBAAAAAAAAAAAAAAAABAEDBgD/xAAkEQAABQMCBwAAAAAAAAAAAAAAAQIDESFB8AQSEzFRcZGh0f/aAAwDAQACEQMRAD8Aqu66edJiLBzJRvbu2cuSklgxL0c1YsAC/XdLKdVBwMWxpeK3uSqsxGfPdAtUnMQ98nDZUiXiGHExWmqQXbv3+jzYbg4mllNORwYOUvTjXfx5fLiEiOo/POIpSolLuzEUqE1emkMxcljiayIa1OxvalJT1wu3gHM3pOTYWmJijTDiMV4KQNKVHskjyTUn7PoWGugtVqHHJk6Ha2XH/9k=" data-deferred="1" jsaction="error:trigger.Ci0Ntd" alt="" class="GRN7Gc" jsname="i1Vy9" data-sz="16" data-iml="765.4000000953674" data-atf="true"><div class="dmeZbb">나무위키</div></div><span class="OztcRd btTgYb cS4Vcb-pGL6qe-mji9Ze" aria-label="하정우 - 나무위키">하정우 - 나무위키</span></a></div>
    '''

    # 3. 이미지 div 태그 수집  
    image_url = []
    for i in range(50) : # image 개수 지정                 
        src = driver.page_source # 현재페이지 source 수집 
        html = BeautifulSoup(src, "html.parser")
        # <div data-ri="0">
        div_img = html.select_one(f'div[data-ri="{i}"]') # 이미지 div tag 1개 수집
    
        # 4. img 태그 수집 & image url 추출
        # <img class="rg_i Q4LuWd", src="url">
        img_tag = div_img.select_one('img[class="rg_i Q4LuWd"]')
        try :
            image_url.append(img_tag.attrs['src'])
            print(str(i+1) + '번째 image url 추출')
        except :
            print(str(i+1) + '번째 image url 없음')
      
            
    # 5. 중복 image url 삭제      
    print('중복 삭제 전 :',len(image_url)) # 43      
    image_url = list(set(image_url)) # 중복 url  삭제 
    print('중복 삭제 후 :', len(image_url)) # 43
    
    ## -- 1차 테스트 --    
    # 6. image 저장 폴더 생성과 이동 
    pwd = r'C:\ITWILL\6_DeepLearning\workspace\chapter07_Selenium_ImageCrawling' # 저장 경로 
    os.mkdir(pwd + '/' + name) # pwd 위치에 폴더 생성(셀럽이름) 
    os.chdir(pwd+"/"+name) # 폴더 이동(현재경로/셀럽이름)
        
    # 7. image url -> image save
    for i in range(len(image_url)) :
        try : # 예외처리 : server file 없음 예외처리 
            file_name = "test"+str(i+1)+".jpg" # test1.jsp
            # server image -> file save
            urlretrieve(image_url[i], filename=file_name)#(url, filepath)
            print(str(i+1) + '번째 image 저장')
        except :
            print('해당 url에 image 없음 : ', image_url[i])        
            
    driver.close() # driver 닫기 
    
    
# 1차 테스트 함수 호출 
#celeb_crawler("하정우")   

# 여러명의 셀럽 이미지 수집 
name_list = ["조인성", "송강호", "전지현"] # 31장, 30장, 30장 

for name in name_list :
    celeb_crawler(name)
    









    