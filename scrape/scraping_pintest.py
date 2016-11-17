
from bs4 import BeautifulSoup
import requests
import feedparser
from selenium import webdriver    #open webdriver for specific browser
from selenium.webdriver.common.keys import Keys   # for necessary browser action
from selenium.webdriver.common.by import By    # For selecting html code
import time
import pandas as pd

#get usernames from Pinterest
def url_scrp(range):
    #set up url to search
    url = 'https://www.pinterest.com/search/users/'
    #use webdriver in order to scrape by page due to the infinite scroll problem
    driver = webdriver.Chrome()
    driver.get(url)
    urls = []
    user_names = []
    #
    for i in xrange(0,range):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)

    visible_links = driver.find_elements_by_class_name('userWrapper')

    for line in visible_links:
        urls.append(line.get_attribute('href'))
    for link in urls:
        user_names.append(link.split('/')[3])
    driver.quit()
    return user_names

def scrape_follow_user_names(user_names, range):
    urls = []
    follower_names = []
    for user_name in user_names:
        url = 'https://www.pinterest.com/' + user_name + '/followers/'
        driver = webdriver.Chrome()
        driver.get(url)

        #
        for i in xrange(0,range):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(10)

        visible_links = driver.find_elements_by_class_name('userWrapper')

        for line in visible_links:
            urls.append(line.get_attribute('href'))
        for link in urls:
            follower_names.append(link.split('/')[3])
        driver.quit()
    return follower_names


if __name__ == '__main__':
    #corresponds to 1000 general users
    feeds = []
    user_names = url_scrp(10)
    follower_names1 = scrape_follow_user_names(user_names[:500],0)
    follower_names2 = scrape_follow_user_names(user_names[500:],0)
    follower_names = follower_names1 + follower_names2
    database = []
    for user in user_names:
        d = feedparser.parse("https://pinterest.com/" + user + "/feed.rss/")
        feeds.append(d)
    for follow in follower_names:
        d = feedparser.parse("https://pinterest.com/" + user + "/feed.rss/")
        feeds.append(d)

    for f in feeds:
        #for pin_info in feed:
        l = len(f['entries'])
        for pin_info in xrange(l):
            soup = BeautifulSoup(f['entries'][pin_info]['summary'])
            pin_description = soup.findAll('p')[1].text
            pin_link = soup.find('a')['href']
            user_id = f['entries'][pin_info]['summary_detail']['base'].split('/')[3]
            entry = (user_id, pin_description, pin_link)
            database.append(entry)

    df = pd.DataFrame(database, columns=["user_id", "pin_description", "pink_link"])
    df.to_csv('user_pins_description.csv', sep='\t', encoding='utf-8')
