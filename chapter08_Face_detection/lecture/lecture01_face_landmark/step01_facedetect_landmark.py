# -*- coding: utf-8 -*-
"""
image 얼굴인식과 68 point landmark 인식

- 필요한 패키지 설치와 68 point landmark data 준비(ppt 참고)
 1. cmake 설치 : dlib 의존성 패키지 설치 
 (tensorflow) >pip install cmake

 2. dlib 설치 : 68 랜드마크 얼굴인식
 (tensorflow) >conda install 파일경로/dlib
    
 3. scikit-image 설치 : image read/save
 (tensorflow) >pip install scikit-image

 4. 68 point landmark data 다운로드    
"""

import dlib # face detection
from skimage import io # image read/save
from glob import glob # dir 패턴검색(*jpg)

# 이미지 파일 경로 지정  
path = r'C:\ITWILL\6_DeepLearning\workspace\chapter08_Face_detection'
image_path = path + "/images" # 작업 대상 image 위치 
croped_path = path + "/images_croped" # croped image 저장 위치 

# dlib 지원하는 도구 
dir(dlib)
'''
get_frontal_face_detector() : 얼굴 인식기(사각점)
shape_predictor() : 68 point landmark(정렬)
image_window() : 이미지 윈도(인식된 이미지 표시)
'''

# 1. 얼굴 인식기(hog 알고리즘)
face_detector = dlib.get_frontal_face_detector()

# 2. 얼굴 68 point landmark 객체 생성 
path = r'C:\ITWILL\6_DeepLearning\tools'
face_68_landmark = dlib.shape_predictor(path+'/shape_predictor_68_face_landmarks.dat')


# 3. 이미지 폴더에서 한 장씩 이미지 인식 
i = 100
for file in glob(image_path+"/*.jpg") : # 폴더에서 순서대로 jpg 파일 읽기 
    image = io.imread(file) # image file 읽기 
    print(image.shape) # image 모양 
    
    # 1) 윈도에 image 표시 
    win = dlib.image_window() # 이미지 윈도 
    win.set_image(image) # 윈도에 원본 이미지 표시 
    
    # 2) image에서 얼굴인식 : object()    
    faces = face_detector(image, 1) # 두번째 인수=1 : 업샘플링 횟수 
    print('인식한 face size =', len(faces)) # 인식된 얼굴 개수 
    
    # 3) 이미지 위에 얼굴 사각점 표시  
    for face in faces : # 1명 -> 1회 반복 
        i += 1 # 카운터 변수 
        win.add_overlay(face) # 얼굴인식 표시
        
        print(face) # 얼굴 사각점 : [(141, 171) (409, 439)]-      
        print(f'왼쪽 : {face.left()}, 위 : {face.top()}, 오른쪽 : {face.right()}, 아래 : {face.bottom()}')
                
        # 2차 : 윈도에 인식된 얼굴 표시 : face 사각점 좌표 겹치기 
        win.add_overlay(face) # 2차 : 이미지 위에 얼굴 겹치기 
        
        # 3차 : face 사각점에서 68 point 겹치기
        face_landmark = face_68_landmark(image, face)
        win.add_overlay(face_landmark) # 3차 : 68 포인트 겹치기 
        
        # 사각점 가져오기 
        rect = face_landmark.rect 
        print('rect : ', rect)
        # rect :  [(67, 80) (175, 187)] 
        print(dir(rect)) # 호출 메서드 
        # top,bottom,left.right
        
        # 4차 : 사각점 좌표 기준 자르기(crop) -> save 
        crop = image[rect.top():rect.bottom(), 
                     rect.left():rect.right()]# [h, w]
        # crop image save
        io.imsave(croped_path + '/croped' + str(i) + '.jpg', crop)#(path/file, object)
        
        
        
        
        
        
        
       



