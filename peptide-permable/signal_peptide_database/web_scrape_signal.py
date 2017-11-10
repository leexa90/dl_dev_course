from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.firefox_binary import FirefoxBinary
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import numpy as np
import time
import os
import pandas as pd
data_file = [x for x in  os.listdir('.') if x.endswith('.html')]
data = pd.DataFrame(columns = ['ID','entry name','protein name','organism','len','status','seq',])
for file in sorted(data_file):
    temp = ''
    f1 = open(file,'r')
    for line in f1:
        temp += line
    text = BeautifulSoup(temp, "html5lib").get_text()
    start = False
    result = []
    for line in text.split('\n'):
        if start:
            result += [line,]
        if 'Accession NumberEntry NameProtein NameOrganismLengthStatusSignal Sequence' in line:
            start = True
    for i in range(0,len(result)-5,1):
        if result[i] != '' and result[i+5] =='confirmed':
            newrow =pd.DataFrame([result[i:i+7]],columns = ['ID','entry name','protein name','organism','len','status','seq',])
            data = data.append( newrow)
    print file,len(data)

data.to_csv('Mammal_signal_peptide.csv',index=0)
data.to_csv('../Mammal_signal_peptide.csv',index=0)
die
import pandas as pd
data = pd.DataFrame(columns=['text'])
import time
for i in range(2,37):
    f1=open(str(i)+'.txt','r')
    for line in f1:
        if 'a href="display_seq.php?details=' in line:
            line1 =  BeautifulSoup(line, "html5lib").get_text().encode('utf-8')
            line2 = line1.split('     ID')
            line2[0] = line2[0].split('IDID')[1]
            data = data.append(line2)
data = data.reset_index(drop=1)
def get_import_stuff(str):
    result = ['',]*12
    str1 = str.split('PEPTIDE SEQUENCE')
    result[0] = str1[0]
    change = ['PEPTIDE NAME','LENGTH','CHIRALITY','LINEAR/CYCLIC','SOURCE','CATEGORY',
              'N TERMINAL MODIFICATION','C TERMINAL MODIFICATION','CHEMICAL MODIFICATION','PUBMED ID','\n']
    for i in range(0,len(change)):
        str1 = str1[1].split(change[i])
        result[i+1] = str1[0]
    return result

impt = ['ID','PEPTIDE SEQUENCE','PEPTIDE NAME','LENGTH','CHIRALITY','LINEAR/CYCLIC','SOURCE','CATEGORY',
        'N TERMINAL MODIFICATION','C TERMINAL MODIFICATION','CHEMICAL MODIFICATION','PUBMED ID']
a=data[0].map(get_import_stuff)
for i in range(len(data)):
    data= data.set_value(i,impt,a[i])
    
data['seq'] = data['PEPTIDE SEQUENCE'].apply(lambda x : x.upper())
data.to_csv('../CPP_DATABSE.csv',index=0)
##for i in range(2,37):
##    page = 'http://crdd.osdd.net/raghava/cppsite/browse_sub1.php?token=Linear&col=5&page='
##    page = page + str(i)
##    driver.get(page)
##    f1= open('%s.txt'%i,'w')
##    f1.write(driver.page_source.encode('utf-8'))
##    f1.close()
def main(i,):
    for year in ['2015',]:
        for month in ['jan','feb','march','april','may','june','july','august',\
                      'sept','oct','nov','dec'][i:i+1]:
            f1=open(month+year+'.txt','w')
            for i in range(1,32):
                print (month,i)
                page = 'www.straitstimes.com/singapore/st-now-news-as-it-happens-%s-%s-%s' %(month,i,year)
                page = 'http://webcache.googleusercontent.com/search?q=cache:' + page
                print page
                # navigate to the application home page
                driver.get(page)
                #driver.implicitly_wait(2)
                P = True
                result = ''
                counter = 0
                time.sleep(2)
                while result == '' and counter <= 5:
                    if counter == 1:
                        driver.get(page)
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
