
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from afinn import Afinn

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk import bigrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from collections import Counter
from Remote_User.models import ClientRegister_Model,Election_model,Election_prediction_model,detection_ratio_model


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            Election_model.objects.all().delete()
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Election_prediction_model.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Election_Tweet_Predicted_Type(request):

    obj =Election_prediction_model.objects.all()
    return render(request, 'SProvider/View_Election_Tweet_Predicted_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_ratio_model.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="Election_Predictions_Results.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Election_prediction_model.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1
        ws.write(row_num, 0, my_row.tweeter, font_style)
        ws.write(row_num, 1, my_row.total_tweet_Time, font_style)
        ws.write(row_num, 2, my_row.tweet, font_style)
        ws.write(row_num, 3, my_row.prediction, font_style)

    wb.save(response)
    return response

def train_model(request):
    se=''
    detection_ratio_model.objects.all().delete()
    df = pd.read_csv('US_election_2020_Tweets.csv')
    df.head()
    df.shape
    # split by speaker
    df_CW = df[df.Tweeted_about == 'Chris Wallace']
    df_JB = df[df.Tweeted_about == 'Vice President Joe Biden']
    df_DT = df[df.Tweeted_about == 'President Donald J. Trump']
    df_CW
    df_DT
    df_JB
    print('Number of segments - Chris Wallace             : ', df_CW.shape[0])
    print('Number of segments - President Donald J. Trump : ', df_DT.shape[0])
    print('Number of segments - Vice President Joe Biden  : ', df_JB.shape[0])
    # convert to strings
    text_CW = " ".join(txt for txt in df_CW.text)
    text_DT = " ".join(txt for txt in df_DT.text)
    text_JB = " ".join(txt for txt in df_JB.text)

    # compare total text lengths
    print('Length of text - Chris Wallace             : ', len(text_CW))
    print('Length of text - President Donald J. Trump : ', len(text_DT))
    print('Length of text - Vice President Joe Biden  : ', len(text_JB))
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                          width=600, height=400,
                          background_color="white").generate(text_CW)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                          width=600, height=400,
                          background_color="white").generate(text_DT)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=500,
                          width=600, height=400,
                          background_color="white").generate(text_JB)
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    # plt.show()
    # standard stopwords
    my_stopwords = set(nltk.corpus.stopwords.words('english'))
    # additional stopwords
    my_stopwords = my_stopwords.union({"'s", "'ll", "'re", "n't", "'ve", "'m"})
    # lower case
    text = text_CW.lower()
    # tokenize text
    words = nltk.word_tokenize(text)
    # remove single characters
    words = [word for word in words if len(word) > 1]
    # remove stopwords
    words = [word for word in words if word not in my_stopwords]
    # count word frequencies
    word_freqs = nltk.FreqDist(words)
    # plot word frequencies
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.title('Word Frequency - Chris Wallace')
    # word_freqs.plot(50)
    my_bigrams = bigrams(words)
    counts = Counter(my_bigrams)
    counts = dict(counts)
    # convert dictionary to data frame
    dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])
    # select only bigrams occuring at least four times
    dcounts = dcounts[dcounts.frequency >= 4]
    # and sort descending
    dcounts = dcounts.sort_values(by='frequency', ascending=False)
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.barh(list(map(str, dcounts.index)), dcounts.frequency)
    plt.title('Most frequent bigrams - Chris Wallace')
    # plt.grid()
    # plt.show()
    # lower case
    text = text_DT.lower()
    # tokenize text
    words = nltk.word_tokenize(text)
    # remove single characters
    words = [word for word in words if len(word) > 1]
    # remove stopwords
    words = [word for word in words if word not in my_stopwords]
    # count word frequencies
    word_freqs = nltk.FreqDist(words)
    # plot word frequencies
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.title('Word Frequency - President Donald J. Trump')
    # word_freqs.plot(50)
    my_bigrams = bigrams(words)
    counts = Counter(my_bigrams)
    counts = dict(counts)
    # convert dictionary to data frame
    dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])
    # select only bigrams occuring at least four times
    dcounts = dcounts[dcounts.frequency >= 4]
    # and sort descending
    dcounts = dcounts.sort_values(by='frequency', ascending=False)
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.barh(list(map(str, dcounts.index)), dcounts.frequency)
    plt.title('Most frequent bigrams - President Donald J. Trump')
    plt.grid()
    # plt.show()
    # lower case
    text = text_JB.lower()
    # tokenize text
    words = nltk.word_tokenize(text)
    # remove single characters
    words = [word for word in words if len(word) > 1]
    # remove stopwords
    words = [word for word in words if word not in my_stopwords]
    # count word frequencies
    word_freqs = nltk.FreqDist(words)
    # plot word frequencies
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.title('Word Frequency - Vice President Joe Biden')
    # word_freqs.plot(50)
    my_bigrams = bigrams(words)
    counts = Counter(my_bigrams)
    counts = dict(counts)
    # convert dictionary to data frame
    dcounts = pd.DataFrame.from_dict(counts, orient='index', columns=['frequency'])
    # select only bigrams occuring at least four times
    dcounts = dcounts[dcounts.frequency >= 4]
    # and sort descending
    dcounts = dcounts.sort_values(by='frequency', ascending=False)
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.barh(list(map(str, dcounts.index)), dcounts.frequency)
    plt.title('Most frequent bigrams - Vice President Joe Biden')
    plt.grid()
    # plt.show()

    sia = SentimentIntensityAnalyzer()


    sent = sia.polarity_scores(text_CW)
    sent_val = sent['compound']
    sent.pop('compound')
    print('CW: sentiment score = ', sent_val)
    print('CW: split = ', sent)

    sent1 = sia.polarity_scores(text_DT)
    sent_val1 = sent1['compound']
    sent1.pop('compound')
    print('DT: sentiment score = ', sent_val1)
    print('DT: split = ', sent1)

    sent2 = sia.polarity_scores(text_JB)
    sent_val2 = sent2['compound']
    sent2.pop('compound')
    print('JB: sentiment score = ', sent_val2)
    print('JB: split = ', sent2)

    obj1 = Election_model.objects.values('tweeter', 'total_tweet_Time','tweet')
    Election_prediction_model.objects.all().delete()
    for t in obj1:
        tweeter = t['tweeter']
        total_tweet_Time = t['total_tweet_Time']
        tweet= t['tweet']

        sentiment_dict = sia.polarity_scores(tweet)

        if sentiment_dict['compound'] >= 0.05:
            # print("Positive")
             se = "Positive"
        elif sentiment_dict['compound'] <= - 0.05:
                # print("Negative")
                se = "Negative"
        else:
            # print("Neutral")
            se = "Neutral"


        Election_prediction_model.objects.create(tweeter=tweeter,total_tweet_Time=total_tweet_Time,tweet=tweet,prediction=se)

    ratio = ""
    kword = 'Positive'
    print(kword)
    obj = Election_prediction_model.objects.all().filter(Q(prediction=kword))
    obj1 = Election_prediction_model.objects.all()
    count = obj.count();
    count1 = obj1.count();
    if count1>0:
        ratio = (count / count1) * 100
    if ratio != 0 and count1!=0:
        detection_ratio_model.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Negative'
    print(kword1)
    obj1 = Election_prediction_model.objects.all().filter(Q(prediction=kword1))
    obj11 = Election_prediction_model.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    if count11>0:
        ratio1 = (count1 / count11) * 100
    if ratio1 != 0 and count11!=0:
        detection_ratio_model.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'Neutral'
    print(kword12)
    obj12 = Election_prediction_model.objects.all().filter(Q(prediction=kword12))
    obj112 = Election_prediction_model.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    if count112>0:
        ratio12 = (count12 / count112) * 100
    if ratio12 != 0 and count112!=0:
        detection_ratio_model.objects.create(names=kword12, ratio=ratio12)

    obj = detection_ratio_model.objects.all()
    return render(request,'SProvider/train_model.html', {'objs': obj,'CW':sent,'DT':sent1,'JB':sent2})














