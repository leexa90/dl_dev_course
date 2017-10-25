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
page = 'https://acc-reg.sphdigital.com/RegAuth2/sphLogin.html?svc=sts&goto=https%3A%2F%2Facc-auth.sphdigital.com%2Famserver%2Foauth2%2Fauthorize%3Fresponse_type%3Dcode%26client_id%3Dstraitstimes_oauth%26redirect_uri%3Dhttps%253A%252F%252Fstraitstimes-mobilegateway-api-SIN-LIVE.stapi.straitstimes.com%252Fapi%252FSphToken%26state%3D%257B%2522originalUrl%2522%253A%2522http%253A%252F%252Fwww.straitstimes.com%252Fsingapore%252Fst-now-news-as-it-happens-may-1-2017%253Flogin%253Dtrue%2522%252C%2522sessionId%2522%253A968011982%252C%2522sessionToken%2522%253A%25228d934b2707924890b355e8176cb35e5e%2522%252C%2522deviceId%2522%253A%2522fa470d7d2fb6446a392d13f04a0abc63103.20.170.81%2522%252C%2522platform%2522%253A%2522web%2522%252C%2522company%2522%253A%2522sph%2522%252C%2522publication%2522%253A%2522ST%2522%252C%2522longitude%2522%253Anull%252C%2522latitude%2522%253Anull%252C%2522newlyRegistered%2522%253A%2522%2522%257D%26scope%3Daonickname%2520uid%2520sn%2520cn%2520aoregservice%2520aovisitorid%2520aologinid'
driver.get(page)

import time
def main(i,):
    for year in ['2017',]:
        for month in ['jan','feb','march','april','may','june','july','august',\
                      'sept','oct','nov','dec'][i:i+1]:
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
                while result == '' and counter <= 5:
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
