# News-sentiment-tracker
News Crawler + Simple Sentiment tracker

## Screenshot of running
<img width="923" alt="Screen Shot 2019-03-20 at 7 37 57 PM" src="https://user-images.githubusercontent.com/32749721/54728058-6cb4e100-4b49-11e9-8637-152f79878214.png">

News Sentiment Tracker
Objects:
There are 3 main objects in this program:
•	Sentiment
•	news_class
•	starter

The Sentiment is the object to analyze the sentiment of articles and update the table which contains all the sentiment values of each crypto currency or stock. It runs by calling function calculate(input). The input should be a dataframe that contains the news information scraped by an instance of object starter. In Sentiment object there are some predefined weights to combine current market sentiment value, previous market sentiment value and current sentiment value of the target together to calculate the one we use to update in the table (overall_market_w, last_senti_w, last_overall_market_w).


The news_class is the parent class for scraping news articles. It has multiple child classes that inherit basic functions of processing the news(get_news and process_news). Each child class is implemented according to the website’s structure with its own scraper function. Some child classes will have function to clean the text called clean_text.

The starter is the object to manage the whole scraping and analyzing process. The initialization of starter creates instances of all the news scrapers and an instance of Sentiment. It runs by calling function scrape.

Process:
In the beginning, we create a starter instance. The initialization of starter will create instances for all child classes of news_class and one instance of Sentiment. Then by calling scrape function of starter, the program will check all the latest news, download them, analyze the sentiment and update the table. 

The while loop of main function in the control file keeps this process running. The control file also saves the result of the analysis as a pickle file, so it allows us to continue this after a break.

Important notes:
If the website makes modification in their web structure, then the scraping might not work and need to be modified according to new structure.

