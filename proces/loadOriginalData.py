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
sys.path.append('..')
from myclass.myobj import Paper,Person

import pymysql.cursors

paperDict=dict()
paperDict_cleaned=dict()
personDict=dict()
DBISDict_cleaned=dict()
class ParseOriginalData:

    def run(self,paperfile,autherfile,coauthorfile,reffile):
        self.readPaper(paperfile)
        self.readAuthor(autherfile)
        self.readCoauthor(coauthorfile)
        self.readRef(reffile)
        print 'done!'

    def readPaper(self, file):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                paperDict[row[0]]=Paper(
                id=row[0],
                title=row[1],
                venue=row[2],
                year=row[3]
                )
    
    def updatePaper(self, file):
         with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                id=row[0]
                title=row[1]
                if title=="": continue
                
                title=util.simplifyStr(title)
                paperDict[id].title=title


    def readAuthor(self, file):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                personDict[row[0]]=Person(
                id=row[0],
                name=row[1]
                )
    
    def readCoauthor(self,file):
        fp = open(file, 'r')
        while 1:
            line = fp.readline()
            if not line:
                break
            content=[c for c in line.split('	') if c]
            author_id=content[1]
            paper_id=content[0][1:]
            rank=int(content[2])

            # print author_id
            # print paper_id
            # print rank

            paper=paperDict[paper_id]
            person=personDict[author_id]
            if paper:
                paper.update_authorId(author_id,rank)
            else:
                print 'no paper '+paper_id
            if person:
                person.update_paper(paper_id,rank)
            else:
                print 'no person '+author_id
        fp.close
    
    def readRef(self,file):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                refferred_id=row[0]
                refferring_id=row[1]

                paper_ed=paperDict[refferred_id]
                paper_ing=paperDict[refferring_id]

                paper_ed.update_referredIds(refferring_id)
                paper_ing.update_referringIds(refferred_id)

class ParseCleanedData:

    def readConf(self, file,ishasv=True):
        confdict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                if ishasv:
                    confdict[content[1][1:].strip()]=content[0].strip()
                else:
                    confdict[content[1].strip()]=content[0].strip()
        return confdict

    def readConf_raw(self, file,ishasv=True):
        confdict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                if ishasv:
                    confdict[content[0].strip()]=content[1][1:].strip()
                else:
                    confdict[content[0].strip()]=content[1].strip()
        return confdict

    def readPaper_Conf(self, file):
        paper_confdict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                paper_confdict[content[0].strip()]=content[1].strip()
        return paper_confdict

    def readPaper_raw(self, file):
        paperdict=dict()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                
                if len(content)>1:
                    paperdict[content[0].strip()]=content[1].strip()
        return paperdict

    def readPaper(self, file):
        paperdict=dict()
        html_parser = HTMLParser.HTMLParser()
        with open(file) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                
                if len(content)>1:
                    paperDict_cleaned[content[0].strip()]=content[1].strip()
                    title=content[1].strip()
                    # if title!='' and title[-1]=='.':
                    #     title=title[:-1]
                    title = html_parser.unescape(title)
                    title=util.simplifyStr(title)
                    paperdict[title]=content[0].strip()
        return paperdict

    def matchTime(self, originalDict,file,filterDict=None):
        currentDict=self.readPaper(file)
        i=0
        counter=0
        for key in originalDict:
            originalTitle=originalDict[key].title
            if originalTitle in currentDict:
                
                i+=1
                if not filterDict: continue
                if currentDict[originalTitle] in filterDict:
                    counter+=1
                    del filterDict[currentDict[originalTitle]]
        
        filterDict_new=filterDict.copy()
        for fkey in filterDict:
            # print fkey
            thisTitle= filterDict[fkey].encode('utf-8')
            if len(thisTitle)< 25: continue
            for okey in originalDict:
                originalTitle=originalDict[okey].title
                if len(originalTitle)< 25: continue
                if abs(len(originalTitle)-len(thisTitle))>40: continue
                # print originalTitle
                if (thisTitle in originalTitle) or (originalTitle in thisTitle):
                    print 'thisTitle:     '+ thisTitle
                    print 'originalTitle: '+ originalTitle
                    counter+=1
                    del filterDict_new[fkey]
                    break

        with open('notmatched.csv', 'wb') as f:
            writer = csv.writer(f)
            for key in filterDict_new:
                writer.writerow([key,paperDict_cleaned[key]])
        print i
        print counter
        print len(currentDict)

    def outputProtentialPapers(self,originalDict,notmatchedfile,paperconffile,paperfile,DBISDictfile,filterDict=None):
        notmatchedIds=set()
        with open(notmatchedfile) as f:
            reader = csv.reader(f)
            for row in reader:
                notmatchedId=row[0]
                notmatchedIds.add(notmatchedId)
        
        focused_paper=[]
        paper_confdict=self.readPaper_Conf(paperconffile)
        paperdict=self.readPaper(paperfile)
        DBIS_cleaned=self.readDBISDict(DBISDictfile)
        for key in originalDict:
            originalTitle=originalDict[key].title
            if originalTitle in paperdict and paper_confdict[paperdict[originalTitle]] in DBIS_cleaned and paperdict[originalTitle] not in notmatchedIds:
                focused_paper.append([paperdict[originalTitle],originalTitle,paper_confdict[paperdict[originalTitle]],DBIS_cleaned[paper_confdict[paperdict[originalTitle]]],originalDict[key].year])
        
        with open('focused.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in focused_paper:
                writer.writerow(row)



    def processDBIS(self,file_cleaned,file_DBIS):
        confdict=self.readConf(file_cleaned)
        outputlist=[]
        with open(file_DBIS) as f:
            while 1:
                line = f.readline()
                if not line:
                    break
                content=[c for c in line.split('	') if c]
                confname=content[1].strip()
                confname=confname.replace(' ','')
                
                if confname in confdict:
                    print confname 
                    content.append(confdict[confname])
                    content.append(confname)
                else:
                    protentiallist=[]
                    for key in confdict:
                        if key in confname:
                            protentiallist.append(confdict[key]+':'+key)
                    outputStr=''
                    for cont in protentiallist:
                        outputStr=outputStr+' '+cont
                    content.append(outputStr)
                outputlist.append(content)
        with open('DBIStoCleaned.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in outputlist:
                writer.writerow(row)

    def readDBISDict(self,file):
        DBIS_cleaned=dict()
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                venue_id=row[2]
                DBIS_cleaned[venue_id]=row[3]
        return DBIS_cleaned
        
    def findDBIS_paper(self,paperconffile,paperfile,DBISDictfile):
        DBIS_paperDict=dict()
        paper_confdict=self.readPaper_Conf(paperconffile)
        paperdict=self.readPaper(paperfile)
        DBIS_cleaned=self.readDBISDict(DBISDictfile)
        counter=0
        for key in paperdict:
            if paper_confdict[paperdict[key]] in DBIS_cleaned:
                counter+=1
                DBIS_paperDict[paperdict[key]]=key
        print counter
        return DBIS_paperDict
    
    def getYears(self,file,label):
        paperYear=dict()
        counter=0
        with open(file) as f:
            reader = csv.reader(f)
            for row in reader:
                print counter
                counter+=1
                papername=row[1]
                try:
                    content=dblp.search(papername)
                    if content==None:
                        time.sleep(20)
                        content=dblp.search(papername)
                    if content.empty: continue
                except:
                    time.sleep(180)
                    content=dblp.search(papername)
                    if content==None:
                        time.sleep(20)
                        content=dblp.search(papername)
                    if content.empty: continue
                paperYear[row[0]]=content
                
                
                if counter%200==0:
                    pickle.dump(paperYear, open('notmatchedpaperyear_'+label+'.dat', "wb"), True)

class ProcessDataBase:
    def __init__(self):
        self.connect = pymysql.Connect(
        host='localhost',
        port=3306,
        user='root',
        passwd='111111',
        db='dblp',
        charset='utf8')
        self.cursor = self.connect.cursor()
        self.selectStr="""SELECT * FROM paper WHERE title = %s and conference = %s """

    def test(self):
        papername="Why do Users Tag? Detecting Users' Motivation for Tagging in Social Tagging Systems."
        papername= self.connect.escape(papername)
        confname='ICWSM'
        confname= self.connect.escape(confname)
        print self.selectStr % (papername,confname)
        self.cursor.execute(self.selectStr % (papername,confname))
        for row in self.cursor.fetchall():
            print row[1]

    def getYear_all(self,paperDict,confDict,paper_confDict):
        outputlist=[]
        counter=0
        i=0
        for key in paperDict:
            print i
            i+=1
            papername=paperDict[key]
            papername= self.connect.escape(papername)
            confname=confDict[paper_confDict[key]]
            confname= self.connect.escape(confname)
            # print papername
            self.cursor.execute(self.selectStr % (papername,confname))
            resultlist=self.cursor.fetchall()
            if len(resultlist)>0:
                print resultlist
                year=resultlist[0][1]
                outputlist.append([key,year])
                counter+=1

        print counter
        print len(paperDict)
        with open('paperwithyear.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in outputlist:
                writer.writerow(row)

    def getYear_DBIS(self,paperDict,confDict,paper_confDict,DBIS_dict):
        outputlist=[]
        counter=0
        counter_dbis=0
        counter_all=0
        for key in paperDict:
            print counter_all
            counter_all+=1
            if paper_confDict[key] in DBIS_dict:
                counter_dbis+=1
                papername=paperDict[key]
                papername= self.connect.escape(papername)
                confname=confDict[paper_confDict[key]]
                confname= self.connect.escape(confname)
            # print papername
                self.cursor.execute(self.selectStr % (papername,confname))
                resultlist=self.cursor.fetchall()
                if len(resultlist)>0:
                    print resultlist
                    year=resultlist[0][1]
                    outputlist.append([key,year])
                    counter+=1

        print counter
        print counter_dbis
        print counter_all
        with open('paperwithyear_dbis.csv', 'wb') as f:
            writer = csv.writer(f)
            for row in outputlist:
                writer.writerow(row)

if __name__ == "__main__":
    # pp=ParseOriginalData()
    # pp.readPaper('E:\\data\\dblp\\base-csv\\paper.csv')
    # pp.updatePaper('./original.csv')
    # with open('original_withvenue.csv', 'wb') as f:
    #     writer = csv.writer(f)
    #     for key in paperDict:
    #         writer.writerow([paperDict[key].id,paperDict[key].title,paperDict[key].year,paperDict[key].venue])
    # pp.run('E:\\data\\dblp\\base-csv\\paper.csv','E:\\data\\dblp\\base-csv\\person.csv','E:\\data\\dblp\\original-data\\AMiner-Coauthor.txt','E:\\data\\dblp\\base-csv\\refs.csv')
    pcd=ParseCleanedData()
    paperdict=pcd.readPaper_raw('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt')
    confdict=pcd.readConf_raw('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_conf.txt')
    paper_confdict=pcd.readPaper_Conf('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt')
    dbis_dict=pcd.readDBISDict('E:\\data\\dblp\\mydata\\DBIStoCleaned.csv')

    # filterdict=pcd.findDBIS_paper('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt','E:\\data\\dblp\\mydata\\DBIStoCleaned.csv')
    # pcd.matchTime(paperDict,'C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt',filterdict)
    # pcd.outputProtentialPapers(paperDict,'./notmatched.csv','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt','E:\\data\\dblp\\mydata\\DBIStoCleaned.csv',)
    # pcd.processDBIS('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_conf.txt','E:\\data\\dblp\\sun\\DBIS\\conf.txt')
    # pcd.getYears(sys.argv[1],str(sys.argv[2]))
    # time.sleep(10)
    # authors = dblp.search('dfahkfhnlkamefl;keakwf;la')
    # print authors.empty
    # print authors["Year"][0]==None
    pdb=ProcessDataBase()
    # pdb.test()
    pdb.getYear_DBIS(paperdict,confdict,paper_confdict,dbis_dict)