from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# create a new Firefox session
caps = DesiredCapabilities().FIREFOX
caps["marionette"] = True
caps["pageLoadStrategy"] = "normal"  #  complete
driver = webdriver.Firefox(capabilities=caps)
driver.maximize_window()
import time
def main():
    for year in ['2016',]:
        for month in ['jan','feb','march','april','may','june','july','august',\
                      'sept','oct','nov','dec'][7:]:
            f1=open(month+year+'.txt','w')
            for i in range(1,32):
                print (month,i)
                page = 'http://www.straitstimes.com/singapore/st-now-news-as-it-happens-%s-%s-%s' %(month,i,year)
                
                
                # navigate to the application home page
                driver.get(page)
                #driver.implicitly_wait(2)
                P = True
                result = ''
                counter = 0
                while result == '' and counter <= 3:
                    if counter >= 1:
                        time.sleep(2)
                    counter += 1
                    for line in driver.page_source.split('\n'):
                        #if 'messi' in line.lower():
                        if P is True:  
                            if '</p>' in line.lower() and '<p>' in line.lower() \
                               and '<a href' in line.lower() and 'str.sg' in line.lower():
                                    result += line.encode('utf-8')
                f1.write('%s_%s.txt\n' %(month,i))
                f1.write(BeautifulSoup(result, "html5lib").get_text().encode('utf-8'))
                f1.write('\n')
            f1.close()
