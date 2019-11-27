from django.shortcuts import render
from django.http import HttpResponse
from .models import Crawl
from django.shortcuts import render
from django.shortcuts import redirect

#packages for python

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import word_tokenize
import re
from nltk.corpus import stopwords
nltk.download('punkt')
#불용어 제거를 위한 모듈
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding,SpatialDropout1D
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.constraints import unit_norm
from sklearn.model_selection import KFold
from keras.models import load_model

#    return HttpResponse("성공")
# Create your views here.
def mainp(request): #첫 화면 템플릿
    return render(request,'first2.html')

def sp(request): #첫 화면 템플릿
    return render(request,'second2.html')

def label0(request):
    datas = Crawl.objects.filter(LABEL=0)
    context={
        "datas":datas
    }
    return render(request,"label00.html",context)

def label1(request):
    datas = Crawl.objects.filter(LABEL=1)
    context={
        "datas":datas
    }
    return render(request,"label11.html",context)

def label2(request):
    datas = Crawl.objects.filter(LABEL=2)
    context={
        "datas":datas
    }
    return render(request,"label22.html",context)

def label3(request): #제출후 메인화면으로
    datas = Crawl.objects.filter(LABEL=3)
    context={
        "datas":datas
    }
    return render(request,"label33.html",context)

def insert_data(request):

    url_address = request.GET['url']

    Crawl.objects.all().delete()

    driver = webdriver.Chrome('/Users/apple1/Downloads/chromedriver')#chrome web driver 버전에 맞게 깔기
    driver.implicitly_wait(3)#웹 자원 로드를 위해 3초 기다려줌
    driver.get(url_address)#url 접근

    #이미지 크롤링----------------------------------------------------------------------------------------------------------
    body = driver.find_element_by_tag_name('body')

    #인기순/작성순 선택할 수 있는 영역 클릭
    #driver.find_element_by_xpath('//paper-button[@class="dropdown-trigger style-scope yt-dropdown-menu"]').click()
    #인기순 카테고리 클릭
    #driver.find_element_by_xpath('//paper-listbox[@class="dropdown-content style-scope yt-dropdown-menu"]/a[1]').click()

    num_pagedown =100 #n번 밑으로 내리는 것
    while num_pagedown:
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(2)
        num_pagedown -=1

    page=driver.page_source
    soup=BeautifulSoup(page,'html.parser')
    #real=soup.find('video')
    #real=real.get('src')
    cmmt_box=soup.find_all('ytd-comment-renderer',attrs={'id':'comment'})

    X=[]
    tokenizer=TreebankWordTokenizer()
    stop_words=set(stopwords.words('english'))
    #이모티콘 제거를 위한 모듈
    emoji_pattern = re.compile("["
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                u"\U00002702-\U000027B0"
                u"\U000024C2-\U0001F251"
                u"\U0001f926-\U0001f937"
                u'\U00010000-\U0010ffff'
                u"\u200d"
                u"\u2640-\u2642"
                u"\u2600-\u2B55"
                u"\u23cf"
                u"\u23e9"
                u"\u231a"
                u"\u3030"
                u"\ufe0f"
            "]+", flags=re.UNICODE)

    for b in cmmt_box :
        comment=b.find('yt-formatted-string',attrs={'id':'content-text'})
        comment=comment.getText().split('\n')[0]
        if comment=='' :continue

        id_name=b.find('a',attrs={'id':'author-text'})
        id_name=id_name.getText().strip()
        if id_name=="":continue

        date=b.find('a',attrs={'class':'yt-simple-endpoint style-scope yt-formatted-string'})
        date=date.getText().split('\n')[0]
        if date=='':continue

        image=b.find('img')
        image=image.get('src')
        #if image=='':continue

        line=str(comment)
        line=line.lower()
        #don't erase [?,:,"]
        line=re.sub('[-=+,#/\^$.@*\※~&%ㆍ!’』\\‘|\(\)\[\]\<\>`“\'…》;]','',line)
        line=re.sub(emoji_pattern,'',line)
        line2=tokenizer.tokenize(line)
        X.append(line2)

        Crawl.objects.create(
            NAME=id_name,
            DATE=date,
            IMG=image,
            CONTENT=comment,

        )

    #새로운 텍스트 정제 후 모델 load-------------------------------------------------------------------------------

    n=WordNetLemmatizer()
    x_data=[]

    for i in X:
        result=[]
        for w in i:
            #표제어 추출
            n.lemmatize(w)
            #불용어 제거
            if w not in stop_words:
                result.append(w)

        x_data.append(result)

    x2_data=[]

    for j in range(len(x_data)):
        line_f=' '.join(x_data[j])
        x2_data.append(line_f)

    num_classes=5
    MAX_WORDS=6000
    MAX_LEN=50
    EMBEDDING_DIM=100

    # loading
    with open('/Users/apple1/youtube_caps/code/tokenizer.pickle', 'rb') as handle:
        t = pickle.load(handle)

    t.fit_on_texts(x2_data)
    x=t.texts_to_sequences(x2_data)
    x2=sequence.pad_sequences(x,maxlen=MAX_LEN)

    model =load_model("/Users/apple1/youtube_caps/model.h5")
    new_y=model.predict_classes(x2)

    i=0
    for each, k in zip(Crawl.objects.all(), new_y):
        each.PRE=x2_data[i]
        each.LABEL=k
        each.save()
        i+=1

    '''def get_num0(self):
        return Crawl.objects.filter(LABEL__contains=0).count()
    def get_num1(self):
        return Crawl.objects.filter(LABEL__contains=1).count()
    def get_num2(self):
        return Crawl.objects.filter(LABEL__contains=2).count()
    def get_num3(self):
        return Crawl.objects.filter(LABEL__contains=3).count()'''

    return render(request,"second2.html",context={"real_video":url_address})
