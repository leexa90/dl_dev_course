f1 = open('pmcXmlp53.txt','r')
from bs4 import BeautifulSoup

page = BeautifulSoup(f1, "lxml")
print len(page.findAll('abstract',limit=10))
##from HTMLParser import HTMLParser
##for line in f1:
##    print HTMLParser().feed(line)
