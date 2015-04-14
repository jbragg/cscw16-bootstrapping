
import sys
import urllib
import urllib2
import mechanize
from bs4 import BeautifulSoup
import csv
import json
import os
import re

YEARS = ['12','13','14']

def get_papers_from_acl(year):

    papers_url = 'http://www.aclweb.org/anthology/P/P' + year
    soup = BeautifulSoup(urllib.urlopen(papers_url).read())
    papers = soup.findAll('i')
    paper_list = []
    for paper in papers:
        if "Invited Talk:" not in paper.text and "Panel:" not in paper.text and "Front Matter" not in paper.text:
            paper_list.append(paper.text)

    return paper_list

def get_citation(paper):

    search_url = 'http://academic.research.microsoft.com/Search?query=\"' + paper + '\"' +' acl' 
    #print search_url
    soup = BeautifulSoup(urllib.urlopen(search_url).read())
    papers = soup.findAll('li', {'class': 'paper-item'})
    
    if not papers:
        search_url = 'http://academic.research.microsoft.com/Search?query=' + paper + '' +' acl'
        soup = BeautifulSoup(urllib.urlopen(search_url).read())
        papers = soup.findAll('li', {'class': 'paper-item'})
        if not papers:
            return 0

    paper = papers[0]
    citation = paper.find('span', {'class': 'citation'})
    if citation:
        citation_count =  citation.text.split(': ')[1].replace(')','')
    else:
        citation_count = 0
    return int(citation_count)


def main():
    fout = open('ACL_paper_citation.csv','a')
    writer = csv.writer(fout)    
    for year in YEARS:
        print "------" + year + "------"
        paper_list = get_papers_from_acl(year)
        count = 0
        for paper in paper_list:
            '''
            if count < 15:
                count+=1
                continue
            '''
            try:
                print paper
                citation_count = get_citation(paper)
                print citation_count
                writer.writerow([year, paper, citation_count])
            except:
                print "something wrong in parsing"
if __name__ == '__main__':
    main()