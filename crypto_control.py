#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:34:15 2018

@author: louis
"""

import datetime
import pickle
import pandas as pd
import numpy as np
#from test_starter_v4 import starter
#from test_starter_v3_debugger import starter

import os.path

from bs4 import BeautifulSoup
import requests
from newspaper import Article

from nltk.tokenize import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict

class Sentiment(object):
    def __init__(self):
        self.index = ['BTC','ETH','EOS','LTC','XRP','BCH',
                 'ETC','XMR','ZEC','QTUM','NEO',
                 'TRX','BTM','KEY','TRUE','XUC','XLM','DASH',
                 'USDT','BTS','BIX','OMG','AST','UBTC','WPR','BNB',
            		'PAX','ZRX','HSR','RVN','ADA','OKB','GTO','TUSD',
            		'BAT','GVT','DOGE','DENT','XIN','MITH','PHX',
            		'BCPT','GNT','SC','IOT','DLT','WTC','PAI','VET','ICX',
            		'NPXS','ONT','STORM','WAVES','AE','WAN','NCASH','LSK',
            		'XEM','ARN','HYDRO','ELF','APIS','IOST','MDA','DOCK','ETF',
            		'XVG','ZIL','QASH','QKC','KNC','TNT','POLY','MBT','HT','MGO',
            		'IOTX','SWFTC','BLZ','BTG','PAL','DNT','INT','MFT','MTH',
            		'BMX','SRN','GO','BCD','NOAH','UBEX','NAS','LINK','ZEN','MCO',
            		'DGD','BBK','ABT','YOYOW']
        self.coin_names = {'bitcoin':'BTC',
                      'ethereum':'ETH',
                      'litecoin':'LTC',
                      'bitcoin cash':'BCH',
                      'ethereum classic':'ETC',
                      'monero':'XMR',
                      'zcash':'ZEC',
                      'tron':'TRX',
                      'bytom':'BTM',
                      'selfkey':'KEY',
                      'true chain':'TRUE',
                      'exchange union':'XUC',
                      'stellar':'XLM',
                      'cardano':'ADA',
                      'Tether':'USDT',
                      'Bitshares':'BTS',
                      'BiboxCoin':'BIX',
                      'OmiseGo':'OMG',
                      'AirSwap':'AST',
                      'UnitedBitcoin':'UBTC',
                      'WePower':'WPR',
                      'Binance Coin':'BNB','Paxos Standard':'PAX','0x':'ZRX','Hshare':'HSR','Ravencoin':'RVN',
     'Cardano':'ADA','Okex':'OKB','GIFTO':'GTO','True USD':'TUSD','Basic Attention Token':'BAT','Genesis Vision':'GVT','Dogecoin':'DOGE','Innity Economics':'XIN',
     'Mithril':'MITH','Red Pulse Phoenix':'PHX','BlockMason Credit Protocol':'BCPT',
			'Golem Network Token':'GNT','Siacoin':'SC','IOTA':'IOT','Agrello Delta':'DLT',
			'Waltonchain':'WTC','Project Pai':'PAI','Vechain':'VET','ICON Project':'ICX',
			'Pundi X':'NPXS','Ontology':'ONT','Aeternity':'AE','Wanchain':'WAN','Nucleus Vision':'NCASH',
			'Lisk':'LSK','NEM':'XEM','Aeron':'ARN','Hydrogen':'HYDRO','aelf':'ELF','IOS token':'IOST',
			'Moeda':'MDA','Dock.io':'DOCK','EthereumFog':'ETF','Verge':'XVG','Zilliqa':'ZIL','Quoine Liquid':'QASH',
			'QuarkChain':'QKC','Kyber Network':'KNC','Tierion':'TNT','Polymath Network':'POLY','Multibot':'MBT',
			'Huobi Token':'HT','MobileGo':'MGO','IoTeX Network':'IOTX','SwftCoin':'SWFTC','Bluzelle':'BLZ',
			'Bitcoin Gold':'BTG','PolicyPal Network':'PAL','district0x':'DNT','Internet Node Token':'INT',
			'Mainframe':'MFT','Monetha':'MTH','BitMart Coin':'BMX','SirinLabs':'SRN','GoChain':'GO',
			'Bitcoin Diamond':'BCD','NOAHCOIN':'NOAH','Nebulas':'NAS','ChainLink':'LINK','Horizen':'ZEN',
			'Crypto.com':'MCO','Digix DAO':'DGD','BitBlocks':'BBK','ArcBlock':'ABT'}
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
        if website in ["thebitcoinnews","newsbtc","cryptoslate","coinstaker","btcwires","bitcoinist","ccn"]:
            self.change_text = True
        else:
            self.change_text = False
        self.k = k
            
    def clean_text(text):
        pass
    
    def get_news(self, links):
        news_text = []
        self.news_latest = links[0]
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
            except Exception as e: 
                print(e)
                pass
        return news_text
    
    def write_into_mysql():
        pass
    
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
        
        
        
        
class thebitcoinnews(news_class):
    def scraper(self):
        links = []
        url = 'https://thebitcoinnews.com/category/bitcoin-news/'
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
        
            for post in soup.find_all('div', class_ = 'td-block-span6'):
                link = post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
                
        
    
    def clean_text(self, texxt):
        content = texxt.split("\n\n")
        keep = len(content)
        if keep < 5:
            return None
        if 'source' == content[-1][:6]:
            keep -= 2
        try:
            for i in range(len(content) - 1, len(content) - 8, -1):
                if content[i] == "For the latest cryptocurrency news, join our Telegram!":
                    keep = i
                    break
        except Exception:
            pass
        return("\n\n".join(content[:keep]))

class newsbtc(news_class):
    def clean_text(self,texxt):
        content = texxt.split("\n\n")
        keep = len(content)
        if 'Featured' == content[-1][:8]:
            keep -= 1
        elif 'Previous' == content[-1][:8]:
            keep -= 2
        return("\n\n".join(content[:keep]))
    
    
    def scraper(self):
        links = []
        urls = ['https://www.newsbtc.com/category/bitcoin','https://www.newsbtc.com/category/crypto/',
               'https://www.newsbtc.com/category/crypto-tech/','https://www.newsbtc.com/category/industry-news/']
        try:
            r = requests.get(urls[self.k])
            soup = BeautifulSoup(r.text, 'html.parser')
        
            posts = soup.find('div', class_ = 'row posts')
            for post in posts.find_all('div', class_ = 'post-content'):
                link = post.find('a', class_ = 'link').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)

class cryptovest(news_class):
    def scraper(self):
        url = 'https://cryptovest.com/tag/bitcoin-news/'
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            links = []
        
            for post in soup.find_all('div','col-12 col-md-6 col-lg-6 p--8 post'):
                link = 'https://cryptovest.com' + post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
        

class cryptoslate(news_class):
    def scraper(self):
        links = []
        try:
            url = 'https://cryptoslate.com/news'
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
        
            for post in soup.find_all('div', class_ = 'post small-post'):
                link = post.find('h2').find("a").get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
    
    def clean_text(self, texxt):
        content = texxt.split("\n\n")
        content = content[:-2]
        if content[-1][-13:] == "CryptoCompare":
            content = content[:-1]
        return("\n\n".join(content))
    

class cointelegraph(news_class):
    def scraper(self):
           base_url = "https://cointelegraph.com"
           links = []
           try:
               r = requests.get(base_url)
               soup = BeautifulSoup(r.text, 'html.parser')
               recent = soup.find('section', id = 'post-content').find('div',class_ = 'row')
               for post in recent.find_all('div',class_='post boxed'):
                   link = post.find('a').get('href')
                   if link == self.news_latest:
                       break
                   else:
                       links.append(link)
               if self.stop:
                   return self.process_news(links)
               
           except Exception as e: print(e)

class coinstaker(news_class):
    def scraper(self):
        url = 'https://www.coinstaker.com'
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            links = []
        
            for post in soup.find_all('div',class_ = 'blogpost'):
                link = post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
                    
    def clean_text(self,texxt):
        content = texxt.strip().split("\n\n")
        keep = len(content)
        idx = keep
        while idx > 0 and idx > (len(content) - 6):
            if 'Read More' == content[idx - 1][:9]:
                keep = idx - 1
            if 'You can also' == content[idx - 1][:12]:
                keep = idx - 1
            if 'Join us' == content[idx - 1][:7]:
                keep = idx - 1
            idx -= 1
        return("\n\n".join(content[:keep]))
        
class coinspeaker(news_class):
    def scraper(self):
        url = 'https://www.coinspeaker.com'
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            links = []
        
            section = soup.find('div',class_ = 'sectionContent')
            for post in section.find_all('div',class_ = 'itemBlock'):
                link = post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)

class coindesk(news_class):
    def scraper(self):
        
        url = "https://www.coindesk.com/"
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            links = []
        
            for post in soup.find_all('div',class_ = "post-info"):
                link = post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)

class ccn(news_class):
    def scraper(self):
        url = "https://www.ccn.com/"
        links=[]
        try:
            headers={'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers = headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            posts = soup.find('div','posts-row')
            for post in posts.find_all('article'):
                link = post.find('a').get('href')
                if link == self.news_latest:
                    break
                else:
                    links.append(link)
                    
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
            
                
    def clean_text(self,texxt):
        content = texxt.split("\n\n")
        content = content[1:-3]
        return("\n\n".join(content))
        
class btcwires(news_class):
    def scraper(self):
          
       url = "https://www.btcwires.com/"   
       try:
           r = requests.get(url)
           soup = BeautifulSoup(r.text, 'html.parser')
           links = []
       
           main_posts = soup.find("div", class_ = "home-post")
           for post in main_posts.find_all("div",class_ = "post-box"):
               try:
                   detail = post.find('div',class_ = 'post-details')
                   link = detail.find('a').get('href')
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
        return("\n\n".join(content[1:keep]))
    

class bitcoinist(news_class):

    def scraper(self):
        links= []
        url = 'https://bitcoinist.com/latest-news'
        try:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
        
            for post in soup.find_all('div', class_ = 'news-content cf'):
                news = post.find('h3', class_ = 'title')
                link = news.find('a').get('href')
                if link == self.news_latest:
                       break
                else:
                       links.append(link)
                
            if self.stop:
                return self.process_news(links)
        except Exception as e: print(e)
    
    def clean_text(self, texxt):
        content = texxt.strip().split("\n\n")
        keep = len(content)
        return("\n\n".join(content[:keep-3]))


class starter(object):
    
    def __init__(self, isNew):
        self.current_time = datetime.datetime.now()
        #self.time_str = self.current_time.strftime("%Y_%m_%d_%H")
        scaper_bitcoinist = bitcoinist(self.current_time,"bitcoinist",True)
        scaper_btcwires = btcwires(self.current_time,"btcwires",True)
        scaper_coindesk = coindesk(self.current_time,"coindesk",True)
        scaper_ccn = ccn(self.current_time,"ccn",True)
        scaper_coindesk = coindesk(self.current_time,"coindesk",True)
        scaper_coinspeaker = coinspeaker(self.current_time,"coinspeaker",True)
        scaper_coinstaker = coinstaker(self.current_time,"coinstaker",True)
        scaper_cointelegraph = cointelegraph(self.current_time,"cointelegraph",True)
        scaper_cryptovest = cryptovest(self.current_time,"cryptovest",True)
        scaper_cryptoslate = cryptoslate(self.current_time,"cryptoslate",True)
        scaper_thebitcoinnews = thebitcoinnews(self.current_time,"thebitcoinnews",True)
        scaper_newsbtc_1 = newsbtc(self.current_time,"newsbtc",True,0)
        scaper_newsbtc_2 = newsbtc(self.current_time,"newsbtc",True,1)
        scaper_newsbtc_3 = newsbtc(self.current_time,"newsbtc",True,2)
        scaper_newsbtc_4 = newsbtc(self.current_time,"newsbtc",True,3)
        self.scrapers = [scaper_bitcoinist,scaper_btcwires,scaper_ccn,scaper_coindesk,
                         scaper_coinspeaker,scaper_coinstaker,scaper_cointelegraph,scaper_cryptovest,
                         scaper_cryptoslate,scaper_thebitcoinnews,scaper_newsbtc_1,scaper_newsbtc_2,
                         scaper_newsbtc_3,scaper_newsbtc_4]            
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
            save_object(st, "scraper_starter_noDB.pkl")
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
            if os.path.isfile('scraper_starter_noDB.pkl'):
                start = pickle.load(open("scraper_starter_noDB.pkl", "rb"))
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
