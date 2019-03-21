#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:56:57 2018

@author: louis
"""

import datetime
import pickle
import os.path
import pandas as pd
#from mysqldb_operate import MySqlCommand
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
from newspaper import Article
from bs4 import BeautifulSoup
import requests



class Sentiment(object):
    def __init__(self):
        self.index = ['SBUX','NXPI','FB','SFIX','JNJ','BRK.A','CNC','AAPL','SFM','CLDR','PLAY','CBS','NBIX',
                      'SBGI','LAUR','NVIDIA','AWK','AOS','SNAP','GTIM','ERIC','TRVG','SPKE','GOOGL']
        self.coin_names = {
                'starbucks':'SBUX','NXP':'NXPI','facebook':'FB','stitch fix':'SFIX','Johnson & Johnson':'JNJ','Berkshire Hathaway':'BRK.A','Centene Corporation':'CNC',
                'apple':'AAPL','Sprouts Farmers Market':'SFM','DowDuPont':'DWDP','Cloudera':'CLDR','Dave & Busters':'PLAY','CBS Corp':'CBS','Neurocrine':'NBIX',
                      'Sinclair Broadcast':'SBGI','Laureate Education':'LAUR','amazon':'AMZN','American Water Works':'AWK',
                      'A. O. Smith':'AOS','snap':'SNAP','Good Times Restaurants':'GTIM','Ericcson':'ERIC',
                      'Trivago':'TRVG','Spark Energy':'SPKE','alphabet':'GOOGL','google':'GOOGL'}
        for i in self.index:
            self.coin_names[i.lower()] = i
        
        self.sentiment = pd.DataFrame(np.empty((len(self.index),6),dtype=object),index = self.index,
                                      columns = ['Last_News_Updated_Time','Current_sentiment','Last_sentiment',
                                                 'Current_Overall_Market_sentiment','Last_Overall_Market_sentiment','Market_Price'])
        self.sentiment.Market_price = 0
        self.sentiment.Last_News_Updated_Time = None
        self.overall_market_w = 0.3
        self.last_senti_w = 0.3
        self.last_overall_market_w = 0.3
        self.overall_market = None
        self.last_overall_market = None
        
    def sentence_tokenizer(self, article_list):
        sent_tokenize_list = []
        for text in article_list:
            text = text.lower()
            sent_tokenize_list.append(sent_tokenize(text))
        return sent_tokenize_list
    
    def calculate(self,data_s):
        news_articles = list(data_s.Text)
        news_sentences_list = self.sentence_tokenizer(news_articles)
        sid = SentimentIntensityAnalyzer()
        vader_sent = defaultdict(list)
        data_s_senti_list = []
        for j in range(len(news_sentences_list)):
            article = news_sentences_list[j]
            text = ' '.join(article)
            show_keys = [key for key in self.coin_names.keys() if key in text]
            tmp_coin_senti = defaultdict(list)
            
            for idx in range(len(article)):
                current_sentence = article[idx]
                sentiment = sid.polarity_scores(current_sentence)['compound']
                #if sentiment is zero, we don't use it
                if sentiment == 0:
                    continue
                if len(show_keys) > 0:
                    for key in show_keys:
                        if key in current_sentence:
                            vader_sent[self.coin_names[key]].append(sentiment)
                            tmp_coin_senti[self.coin_names[key]].append(sentiment)
                            step = idx + 1
                            #check next 3 sentences' sentiments and add them to the current key.
                            while step < len(article) and step - idx < 4:
                                current_sentence = article[step]
                                sentiment = sid.polarity_scores(current_sentence)['compound']
                                #if sentiment is zero, we don't use it
                                if sentiment == 0:
                                    step += 1
                                    continue
                                vader_sent[self.coin_names[key]].append(sentiment)
                                tmp_coin_senti[self.coin_names[key]].append(sentiment)
                                step += 1
                        else:
                            vader_sent['overall_market'].append(sentiment)
                            tmp_coin_senti['overall_market'].append(sentiment)
                else:
                    vader_sent['overall_market'].append(sentiment)
                    tmp_coin_senti['overall_market'].append(sentiment)
                
            mean_dict = defaultdict(float)
            for k, v in tmp_coin_senti.items():
                mean_dict[k] = float('%.3f'%np.array(v).mean())
            data_s_senti_list.append(mean_dict)
        data_s['Sentiment'] = np.array(data_s_senti_list)                    
        output = defaultdict(float)
        #mean of each sentences
        for key, value in vader_sent.items():
            output[key] = float('%.3f'%np.array(value).mean())
        #mean of each articles
# =============================================================================
#         for key, value in vader_sent.items():
#              output[key] = float('%.3f'%(np.array(value).sum()/len(news_sentences_list))) #too big!
# =============================================================================
        
            
        #update overall market sentiment
        self.last_overall_market = self.overall_market
        if self.last_overall_market:
            self.overall_market = float('%.3f'%(output['overall_market'] * (1-self.last_overall_market_w) + self.last_overall_market * self.last_overall_market_w))
        else:
            self.overall_market = output['overall_market']
        output.pop('overall_market', None)
        return output, data_s
    
    def senti_analyzer(self,data_list):
        flat_list = [item for sublist in data_list for item in sublist]
        data = pd.DataFrame(flat_list,columns = ['Title','Text','Date','Source'])
        the_time = data.Date[0]
        #data_s =  data.sort_values(by = "Date")
        data = data.reset_index(drop=True)
        current_sentiment_dict, news_data = self.calculate(data)
        #update columns of overall market
        self.sentiment.Current_Overall_Market_sentiment = self.overall_market
        self.sentiment.Last_Overall_Market_sentiment = self.last_overall_market
        for idx in self.sentiment.index:
            # if there are new articles about this coin, update it with new sentiment
            
            last_senti = self.sentiment.loc[idx]
            
            if idx in current_sentiment_dict.keys():
                value = current_sentiment_dict[idx]
                if last_senti[1] == None:
                    curr_sentiment = float('%.3f'%(value * (1-self.overall_market_w) + self.overall_market_w * self.overall_market))
                    self.sentiment.loc[idx,:3] = np.array([the_time, curr_sentiment, None])
                else:
                    #update current sentiment
                    t = float(value) * float(1- self.last_senti_w)
                    tt = float(last_senti[1]) * float(self.last_senti_w)
                    curr_sentiment_weighted = t + tt
                    curr_sentiment = float('%.3f'%(curr_sentiment_weighted * (1-self.overall_market_w) + self.overall_market_w * self.overall_market))
                    self.sentiment.loc[idx,:3] = np.array([the_time, curr_sentiment, float(last_senti[1])])
            else:
                #If the coin doesn't show up in these recent articles then update it with overall market sentiment
                #If it has been assigned
                if last_senti[1]:
                    #if it was assigned with overall market sentiment, then again fill it with market sentiment
                    if float(last_senti[1]) == last_senti[4]:
                        self.sentiment.loc[idx,1:3] = np.array([self.overall_market, float(last_senti[1])])
                    #if it was different from overall market value, then we can update it with the affect of overall market sentiment
                    else:
                        t = float(last_senti[1]) * float(1-self.overall_market_w)
                        tt = float(self.overall_market_w)*float(self.overall_market)
                        curr_sentiment = float('%.3f'%(t + tt))
                        self.sentiment.loc[idx,1:3] = np.array([curr_sentiment, float(last_senti[1])])
                #If it has never been assigned, use market sentiment to fill
                else:
                    self.sentiment.loc[idx,1:3] = np.array([self.overall_market, last_senti[1]])
        
        return news_data
                
class news_class():
    def __init__(self, current_time, website, stop, k=None):
        self.news_latest = ""
        self.current_time = current_time
        self.is_exist = False
        self.website = website
        #stop 10s after first scrape
        self.freq1 = 10
        #stop 30s
        self.freq2 = 30
        self.stop = stop
        self.news_latest = None  
        self.news_latest_backup = None
        if website in ["marketwatch","thebitcoinnews","newsbtc","cryptoslate","coinstaker","btcwires","bitcoinist","ccn"]:
            self.change_text = True
        else:
            self.change_text = False
        self.k = k
            
    def clean_text(text):
        pass
    
    def get_news(self, links):
        news_text = []
        self.news_latest = links[0]
        #print("Lastest news is {}".format(self.news_latest))
        for link in links:
            date = self.current_time
            try:
                article = Article(link.strip())
                article.download()
                article.parse()
                text = article.text
                if self.change_text:
                    text = self.clean_text(text)
                if text:
                    news_text.append([article.title,text,date,self.website])
            except Exception: 
                pass
        return news_text
    
    def scraper(self, stop):
        pass
    
    
    def process_news(self, links):
        if len(links) > 0:
            news = self.get_news(links)
            if self.stop:
                print("{} Webscraper started time:{}\nWrote {} news".format(self.website, self.current_time ,len(news)))
                return news
            else:
                self.write_into_csv(news)
                print("{} Webscraper started time:{}\nWrote {} news".format(self.website, self.current_time ,len(news)))
        
        

class seekingalpha(news_class):
    def scraper(self):
          
       url = "https://seekingalpha.com/market-news/all"   
       
       try:
           r = requests.get(url)
           soup = BeautifulSoup(r.text, 'html.parser')
           links = []
       
           main_posts = soup.find("ul", class_ = "mc-list")
           for post in main_posts.find_all("li",class_ = "mc"):
               try:
                   detail = post.find('div',class_="title")
                   link = "https://seekingalpha.com" + detail.find('a').get('href')
                   
                   if link == self.news_latest or link == self.news_latest_backup:
                       break
                   else:
                       links.append(link)
                       
               except:
                   continue
            
           if self.stop:
               return self.process_news(links)
       except Exception as e: print(e)
       
    def get_news(self, links):
        news_text = []
        self.news_latest_backup = self.news_latest
        self.news_latest = links[0]
        #print("Lastest news is {}".format(self.news_latest))
        for link in links:
            date = self.current_time
            try:
                article = Article(link.strip())
                article.download()
                article.parse()
                text = article.text
                if self.change_text:
                    text = self.clean_text(text)
                if text:
                    news_text.append([article.title,text,date,self.website])
            except Exception: 
                pass
        return(news_text)


class motelyfool(news_class):
    def scraper(self):
          
       url = "https://www.fool.com/investing-news/?page=1"   
       try:
           r = requests.get(url)
           soup = BeautifulSoup(r.text, 'html.parser')
           links = []
       
           main_posts = soup.find("div", class_ = "list-content")
           for post in main_posts.find_all("div",class_ = "text"):
               try:
                   link = "https://www.fool.com" + post.find('a').get('href')
                   if link == self.news_latest:
                       break
                   else:
                       links.append(link)
               except:
                   continue
            
           if self.stop:
               return self.process_news(links)
       except Exception as e: print(e)

class marketwatch(news_class):
    def scraper(self):
          
       url = "https://www.marketwatch.com/newsviewer"   
       try:
           r = requests.get(url)
           soup = BeautifulSoup(r.text, 'html.parser')
           links = []
       
           main_posts = soup.find("div", class_ = "nviewer")
           for post in main_posts.find_all('div','nv-text-cont'):
               try:
                   link = "https://www.marketwatch.com" + post.find('a').get('href')
                   if link == self.news_latest:
                       break
                   else:
                       links.append(link)
               except:
                   continue
            
           if self.stop:
               return self.process_news(links)
       except Exception as e: print(e)
       
    def clean_text(self, texxt):
        content = texxt.strip().split("\n\n")
        keep = len(content)
        if content[-1][-5:] == "here.":
            keep -= 1
        return("\n\n".join(content[:keep]))

class investopedia(news_class):
    def scraper(self):
          
       url = "https://www.investopedia.com/news/"   
       try:
           r = requests.get(url)
           soup = BeautifulSoup(r.text, 'html.parser')
           links = []
       
           main_posts = soup.find("ol", class_ = "list gaEvent")
           for post in main_posts.find_all("h3",class_ = "item-title"):
               try:
                   link = "http://www.investopedia.com" + post.find('a').get('href')
                   if link == self.news_latest:
                       break
                   else:
                       links.append(link)
               except:
                   continue
            
           if self.stop:
               return self.process_news(links)
       except Exception as e: print(e)
        
    

class starter(object):
    
    def __init__(self, isNew):
        self.current_time = datetime.datetime.now()
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")
        scaper_investopedia = investopedia(self.current_time,"investopedia",True)
        scaper_motelyfool = motelyfool(self.current_time,"motelyfool",True)
        scaper_marketwatch = marketwatch(self.current_time,"marketwatch",True)
        scaper_seekingalpha = seekingalpha(self.current_time,"seekingalpha",True)
        
        self.scrapers = [scaper_investopedia,scaper_motelyfool,scaper_marketwatch,scaper_seekingalpha]            
        #self.scrapers = [scaper_bitcoinist,scaper_btcwires]
        self.curr_sentiment = Sentiment()
    

    
    def report_start(self):
        self.current_time = datetime.datetime.now()
        print("______________________________")
        print("Sentiment analyzer starts on {}".format(self.current_time))
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")      

    def scrape(self):
        total_news = []
        for cur_scraper in self.scrapers:
            cur_scraper.current_time = self.current_time
            news = cur_scraper.scraper()
            if news != None and len(news) > 0:
                total_news.append(news)
        
        if len(total_news) > 0:
            news_df = self.curr_sentiment.senti_analyzer(total_news)
            with pd.option_context('display.max_rows', None, 'display.max_columns',None,'display.width',5000):
                print(news_df.iloc[:,[0,2,4]])
                
            with pd.option_context('display.max_rows', None, 'display.max_columns',None,'display.width',5000):
                print(self.curr_sentiment.sentiment)
            return 1
        else:
            print("No valid news to update.")
            return 0
        
        
            
    def report_end(self):
        self.current_time = datetime.datetime.now()
        print("______________________________")
        print("Sentiment analyzer ends on {}".format(self.current_time))
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")      

        

def main(start):
    if start:
        st = start
    else:
        st = starter(True)

    while 1:
        c_time = datetime.datetime.now()
        st.report_start()
        isUpdated = st.scrape()
        
        if isUpdated:
            save_object(st, "stock_scraper_starter_noDB.pkl")
        #time.sleep(900)
        st.report_end()
        now = datetime.datetime.now()
        print("Time spent:{}".format(now - c_time))




def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    output.close()
    print("Scraper saved successfully!")


if __name__ == '__main__':
    print("To stop the script execution type CTRL-C")
    while 1:
        try:
            if os.path.isfile('stock_scraper_starter_noDB.pkl'):
                start = pickle.load(open("stock_scraper_starter_noDB.pkl", "rb"))
            else:
                start = None
        except:
            start = None
        try:
            main(start)
        except KeyboardInterrupt:
            resume = input('If you want to continue type the letter c:')
            if resume != 'c':
                break
