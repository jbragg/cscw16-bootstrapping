
import sys
import urllib
import urllib2
import mechanize
from bs4 import BeautifulSoup
import csv
import json
import os
import re

#http://academic.research.microsoft.com/RankList?entitytype=2&topDomainID=2&subDomainID=5&last=0&start=1&end=100


def main():
    fout = open('authors_publication.csv','a')
    writer = csv.writer(fout) 

    start = 1
    end = 100
    for i in range(2661):
        print "-----" + str(i) + "-----"
        authors_url = "http://academic.research.microsoft.com/RankList?entitytype=2&topDomainID=2&subDomainID=5&last=0&start="+str(start)+"&end="+str(end)
        soup = BeautifulSoup(urllib.urlopen(authors_url).read()) 
        #print soup
       
        authors = soup.findAll('div',{'class':'content-narrow'})
        for author in authors:
            try:
                publications = 0
                citations = 0
                spans = author.findAll('span')
                for span in spans:
                    if 'Publications' in span.text:
                        publications = span.text.split(': ')[1]
                    elif 'Citations' in span.text:
                        citations = span.text.split(': ')[1]


                name = author.find('div',{'class':'title'}).text.strip().encode('ascii','ignore')
                print name, publications, citations
                writer.writerow([name,publications,citations])
            except:
                print "something wrong in parsing"
        start+=100
        end+=100        
if __name__ == '__main__':
    main()