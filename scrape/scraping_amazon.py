from bs4 import BeautifulSoup
import requests
import feedparser
from selenium import webdriver    #open webdriver for specific browser
from selenium.webdriver.common.keys import Keys   # for necessary browser action
from selenium.webdriver.common.by import By    # For selecting html code
import time
import urllib
import pandas as pd


# visible_links = driver.find_elements_by_css_selector("a.a-size-base.a-spacing-base.a-spacing-top-base.a-align-center.a-link-normal")
# price_links = driver.find_elements_by_css_selector('span.a-size-large.a-color-secondary.price-display.a-text-bold.a-nowrap')


def amazon_scrp(interval, url):
    product_description = []
    product_price = []
    product_id = []
    database = []
    count = 0

    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(9)
    for i in xrange(0,interval):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        for page in xrange(0+(i*2),(i*2)+2):
            for im in xrange(1,25):
                products = driver.find_element_by_xpath('//*[@id="infinite-scroll-page-' + str(page) + '"]/div[' + str(im) + ']/div[2]/a')
                product_description = products.get_attribute('title')
                product_amazon_link = products.get_attribute('href')
                image = driver.find_element_by_xpath('//*[@id="infinite-scroll-page-' + str(page) + '"]/div[' + str(im) + ']/div[2]/a/img')
                image_url = image.get_attribute('src')
                image_id = image_url.split('/')[5].split('.')[0]
                price = driver.find_element_by_xpath('//*[@id="infinite-scroll-page-' + str(page) + '"]/div[' + str(im) + ']/div[3]/span')
                product_price = price.text
    #product_id.append(image_id)
                entry = (image_id, product_description, product_price[1:], image_url, product_amazon_link)
                database.append(entry)
                print len(database)
    #save figure
                resource = urllib.urlopen(image_url)
                output = open("./images/" + image_id + ".jpg","wb")
                output.write(resource.read())
                output.close()
                print "saved image ..."
    driver.quit
    time.sleep(10)

    return database

if __name__ == '__main__':
    url_woman = 'https://www.amazon.com/gp/gift-finder?ageGroup=woman'
    url_man = 'https://www.amazon.com/gp/gift-finder?ageGroup=man'
    url_teen = 'https://www.amazon.com/gp/gift-finder?ageGroup=teen'
    database1 = amazon_scrp(129,url_woman)
    database2 = amazon_scrp(110,url_man)
    database3 = amazon_scrp(77,url_teen)
    total_database = database1 + database2 + database3
    df = pd.DataFrame(total_database, columns=["image_id", "product_description", "product_price", "image_url","product_amazon_link"])
    df = df.drop_duplicates(subset='image_id')
    df.to_csv('amazon_products_description.csv', sep='\t', encoding='utf-8')
