# -*- coding: utf-8 -*-
import os
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import util
import re
import csv
import time
import HTMLParser
import dblp
import cPickle as pickle
import collections
sys.path.append('.')
from myclass.myobj import Paper,Person,Author

import pymysql.cursors

paperDict=dict()
paperDict_cleaned=dict()
personDict=dict()
DBISDict_cleaned=dict()
class ParseOriginalData:

    startYear=1980
    endYear=2017

    def __init__(self,paperfile,conffile,authorfile,paper_conffile,paper_authorfile,paper_yearfile,focused_conffile):
        self.paperDict_raw=self.readPaper_raw(paperfile)
        self.confDict_raw=self.readConf_raw(conffile)
        self.authorDict_raw=self.readAuthor_raw(authorfile)
        self.paper_confDict=self.readPaper_conf(paper_conffile)
        self.paper_authorDict=self.readPaper_author(paper_authorfile)
        self.paper_yearDict=util.read_csv_withdict(paper_yearfile,0,1)
        self.focused_confDict,self.focused_confDict_area=self.readDBISDict(focused_conffile)

        self.paperDict_filtered=dict()
        self.paperDict_obj=dict()
        self.authorDict_obj=dict()

    def run(self,isWrite):
        self.prepocess()

        self.readPaper_obj()
        self.readAuthor_obj()

        self.staticByyear(self.paperDict_obj,'E:\\data\\dblp\\mydata\\yearCount.dat')
        if isWrite:
            pickle.dump(self.paperDict_obj, open('paperDict_obj.dat', "wb"), True)
            pickle.dump(self.authorDict_obj, open('authorDict_obj.dat', "wb"), True)
        print 'done!'
        
    # To filter these same paper
    def prepocess(self):
        for key in self.paperDict_raw:
            if key not in self.paper_yearDict: continue
            title=self.paperDict_raw[key]
            keywords=title.split(' ')
            if len(keywords)<3: continue
            conf=self.paper_confDict[key]
            if conf not in self.focused_confDict: continue
            newkey=(title, conf)
            self.paperDict_filtered[newkey]=key
        print len(self.paperDict_filtered)
        print len(self.paperDict_raw)


    def readDBISDict(self,file):
        DBIS_cleaned=dict()
        areaDict=dict()
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                venue_id=row[2]
                DBIS_cleaned[venue_id]=row[3]
                areaDict[venue_id]=row[4]
        return DBIS_cleaned,areaDict


    def readConf_raw(self, file,ishasv=True):
        confDict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                if ishasv:
                    confDict[content[0].strip()]=content[1][1:].strip()
                else:
                    confDict[content[0].strip()]=content[1].strip()
        return confDict

    def readPaper_conf(self, file):
        paper_confDict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                paper_confDict[content[0].strip()]=content[1].strip()
        return paper_confDict

    def readPaper_author(self, file):
        paper_authorDict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                if len(content)<2: continue
                paperKey=content[0].strip()
                authorKey=content[1].strip()
                if paperKey in paper_authorDict:
                    authorlist=paper_authorDict[paperKey]
                    if authorKey not in authorlist:
                        authorlist.append(authorKey)
                        paper_authorDict[paperKey]=authorlist
                else:
                    paper_authorDict[paperKey]=[authorKey]

        return paper_authorDict

    def readPaper_raw(self, file):
        paperDict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]                
                if len(content)>1:
                    paperDict[content[0].strip()]=content[1].strip()
        return paperDict

    def readAuthor_raw(self, file):
        authorDict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                
                if len(content)>1:
                    authorDict[content[0].strip()]=content[1].strip()
        return authorDict

    def readPaper_obj(self):
        for key in self.paperDict_filtered:
            paperKey=self.paperDict_filtered[key]
            self.paperDict_obj[paperKey]=Paper(
                id=paperKey,
                title=self.paperDict_raw[paperKey],
                venue=self.paper_confDict[paperKey],
                year=int(self.paper_yearDict[paperKey]),
                authors=self.paper_authorDict[paperKey],
                area=self.focused_confDict_area[self.paper_confDict[paperKey]]
                )

    def countAuthors(self):
        focusedAuthors=dict()
        print len(self.paperDict_obj)
        for key in self.paperDict_obj:
            paper=self.paperDict_obj[key]
            authors=paper.authors
            for author in authors:
                if author not in focusedAuthors:
                    focusedAuthors[author]=1
                else:
                    focusedAuthors[author]=focusedAuthors[author]+1

        util.write_csv_inlist('authorCount.csv',util.dict2list(focusedAuthors))
        return focusedAuthors

    def readAuthor_obj(self):
        for key in self.paperDict_obj:
            paper=self.paperDict_obj[key]
            authors=paper.authors
            author_rank=0
            for author in authors:
                if author in self.authorDict_obj:
                    author_obj=self.authorDict_obj[author]
                    author_obj.update_paper(key, author_rank)
                else:
                    author_obj=Person(id=author, name=self.authorDict_raw[author])
                    author_obj.update_paper(key, author_rank)
                self.authorDict_obj[author]=author_obj
                author_rank=author_rank+1
        for key in self.authorDict_obj:
            author=self.authorDict_obj[key]
            author.update_byyear(self.startYear,self.endYear,self.paperDict_obj)
            author.countArea(self.startYear,self.endYear,self.paperDict_obj)

    def staticByyear(self,inputDict,outputfile):
        yearCount=collections.OrderedDict()
        for i in range (1960,2017):
            yearCount[i]=0
        for key in inputDict:
            paper=self.paperDict_obj[key]
            theyear=paper.year
            if theyear in yearCount:
                yearCount[theyear]=yearCount[theyear]+1
        for key in yearCount:
            print str(key)+' '+str(yearCount[key])
        pickle.dump(yearCount, open(outputfile, "wb"), True)



if __name__ == "__main__":
    pod=ParseOriginalData('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_author.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_author.txt','E:\\data\\dblp\\mydata\\paperwithyear_dbis.csv','E:\\data\\dblp\\mydata\\DBIStoCleaned_20_area.csv')
    # pod.staticByyear()
    pod.run(True)
    # inputDict=dict()
    # with open('havelist.csv') as f:
    #     reader = csv.reader(f)
    #     next(reader)
    #     for row in reader:
    #         inputDict[row[1]]=row[2]
    # pod.staticByyear(inputDict,'yearCount_aminer.dat')
