# -*- coding: utf-8 -*-
"""
step03_css_selector.py
"""

from selenium import webdriver # driver 생성  
from selenium.webdriver.common.by import By 
import time # 일지 정지 


# 태그 수집 함수 
def keyword_search(keyword) :
    # 1. driver 객체 생성 
    path = r"C:\ITWILL\6_DeepLearning\tools\chromedriver_win32" 
    driver = webdriver.Chrome(path + '/chromedriver.exe')
    
    
    # 2. url 이동 : naver news 검색 페이지 이동
    url = f'https://search.naver.com/search.naver?where=news&ie=utf8&sm=nws_hty&query={keyword}'
    driver.get(url) # 검색 페이지 이동   
    time.sleep(3)
    
    
    # 3. 선택자 지정 
    links = driver.find_elements(By.CSS_SELECTOR, 'a.news_tit') # 'tag.class'
    #links = driver.find_elements(By.TAG_NAME, 'a') # 'tag'
    print('수집 a_tags 개수 =', len(links)) # 10개 : list
    
    # element에서 호출가능 속성 & 메서드 확인 
    print(dir(links[0]))
    '''
    element.click() : 클릭 효과 
    element.find_element() : 하위 태그 반환 
    element.get_attribute('속성') : 속성값 반환 
    element.text : 내용 반환 
    '''
    
    # 4. urls 추출 
    urls = []
    contents = [] # 추가 
    for a in links :          
        contents.append(a.text) # 추가 : tag 내용 
        # element.get_attribute('속성')  
        urls.append(a.get_attribute('href')) # url 
    
    driver.close() # 창 닫기 
    
    return urls, contents


# 함수 호출 
keyword = input('검색어 입력 : ') # 파이썬, 크롤링  
urls, contents = keyword_search(keyword)
print(urls)
'''
['http://www.datanews.co.kr/news/article.html?no=127730', 
 'http://www.newsis.com/view/?id=NISX20230525_0002317056&cID=10403&pID=15000', 
 'http://www.naeil.com/news_view/?id_art=462642', 'https://byline.network/?p=9004111222505957', 'http://www.newsis.com/view/?id=NISX20230602_0002325992&cID=14001&pID=14000', 'https://www.kmaeil.com/news/articleView.html?idxno=401431', 'https://it.chosun.com/site/data/html_dir/2023/05/28/2023052800266.html', 'https://www.ajunews.com/view/20230530150143595', 'http://www.veritas-a.com/news/articleView.html?idxno=459155', 'http://www.wsobi.com/news/articleView.html?idxno=202138']
'''
print(contents)





