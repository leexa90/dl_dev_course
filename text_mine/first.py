from Bio import Entrez

import urllib 
import urllib2
import sys

def fetchByQuery(query,days):
    Entrez.email = "leexa@bii.a-star.edu.sg" # you must give NCBI an email address
    searchHandle=Entrez.esearch(db="pmc", reldate=days, term=query, usehistory="y")
    searchResults=Entrez.read(searchHandle)
    searchHandle.close()
    webEnv=searchResults["WebEnv"]
    queryKey=searchResults["QueryKey"]
    batchSize=10
    try:
        fetchHandle = Entrez.efetch(db="pmc", retmax=1000000000, retmode="xml", webenv=webEnv, query_key=queryKey)
        data=fetchHandle.read()
        fetchHandle.close()
        return data
    except:
        return None

days=10000 #looking for papers in the last 100 days
termList=["tpp","riboswitch"]
termList=["p53"] 
terms = ''
for i in termList:
    terms += i
query=" AND ".join(termList)
xml_data=fetchByQuery(query,days)
if xml_data==None: 
    print 80*"*"+"\n"
    print "This search returned no hits"

else:
    f=open("pmcXml%s.txt" %terms ,"w")
    f.write(xml_data)
    f.close()
