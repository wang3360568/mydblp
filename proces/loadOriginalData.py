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


class ParseOriginalData:

    def __init__(self):
        self.paperDict=dict()
        self.paperDict_cleaned=dict()
        self.personDict=dict()
        self.DBISDict_cleaned=dict()

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
                self.paperDict[row[0]]=Paper(
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
                self.paperDict[id].title=title


    def readAuthor(self, file):
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.personDict[row[0]]=Person(
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

            paper=self.paperDict[paper_id]
            person=self.personDict[author_id]
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

                paper_ed=self.paperDict[refferred_id]
                paper_ing=self.paperDict[refferring_id]

                paper_ed.update_referredIds(refferring_id)
                paper_ing.update_referringIds(refferred_id)

    def reverseDict(self):
        paperTitleDict=dict()
        for key in self.paperDict:
            paperTitleDict[self.paperDict[key].title]=key

        return paperTitleDict


    def matchAminerAndDBLP(self, focusedPaperDict, filterDict=None):
        haveCounter=0
        havelist=[]
        nolist=[]
        html_parser = HTMLParser.HTMLParser()
        paperTitleDict=self.reverseDict()
        for key in focusedPaperDict:
            originalTitle = html_parser.unescape(focusedPaperDict[key].title)
            originalTitle=util.simplifyStr(originalTitle)
            # print originalTitle
            if originalTitle in paperTitleDict:
                havelist.append([paperTitleDict[originalTitle],key,focusedPaperDict[key].title])
                haveCounter+=1
                print haveCounter
            else:
                nolist.append([key,focusedPaperDict[key].title])
        
        util.write_csv_inlist('havelist.csv',havelist,['aminerKey','dblpKey','title'])
        util.write_csv_inlist('nolist.csv',nolist,['aminerKey','title'])
        print haveCounter
        print len(focusedPaperDict)

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

    def findDBIS_paper_withfilter(self,paper_confdict,paperdict,DBIS_cleaned,myfilter=None):
        DBIS_paperlist=list()
        counter=0
        for key in paperdict:
            confkey=paper_confdict[key]
            if confkey in DBIS_cleaned:
                counter+=1
                if key not in myfilter:
                    DBIS_paperlist.append([key, paperdict[key]])
        print counter
        util.write_csv_inlist('notmatched.csv',DBIS_paperlist)

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



class Statistics:
    def __init__(self,focusedDict,paper_confDict,confDict,DBIS_dict):
        self.focusedDict=focusedDict
        # self.paperDict=paperDict
        self.paper_confDict=paper_confDict
        self.confDict=confDict
        self.DBIS_dict=DBIS_dict
        self.year_start=1990
        self.year_end=2016
        self.yearDict=dict()
        confDict_dbis=dict()
        for conf in DBIS_dict:
            confDict_dbis[conf]=0
        for year in range(self.year_start,self.year_end):
            newconfDict=confDict_dbis.copy()
            self.yearDict[str(year)]=newconfDict
    
    def yearDistribution(self):
        for key in self.focusedDict:
            thisYear=self.focusedDict[key]
            if thisYear in self.yearDict:
                conf=self.paper_confDict[key]
                if conf not in self.DBIS_dict: continue
                self.yearDict[thisYear][conf]=self.yearDict[thisYear][conf]+1
    
    def output(self):
        outputlist=[]
        for year in self.yearDict:
            confDict_oneyear=self.yearDict[year]
            for conf in confDict_oneyear:
                outputlist.append([year,conf,self.confDict[conf],confDict_oneyear[conf]])
        
        util.write_csv_inlist('statistic_conf_byyear.csv',outputlist)


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
        self.selectStr_easy="""SELECT * FROM paper WHERE title = %s """

    def test(self):
        papername="Space-optimal heavy hitters with strong error bounds."
        papername= self.connect.escape(papername)
        confname='PODS'
        confname= self.connect.escape(confname)
        print self.selectStr % (papername,confname)
        self.cursor.execute(self.selectStr % (papername,confname))
        for row in self.cursor.fetchall():
            print row[1]
    
    def selectData_twokey(self,outputfile,key1,key2):
        selectStr="""SELECT * FROM paper WHERE conference like %s or conference like %s """
        key1= self.connect.escape(key1)
        key2= self.connect.escape(key2)
        self.cursor.execute(selectStr % (key1,key2))
        outputlist=[]
        for row in self.cursor.fetchall():
            outputlist.append([row[0].encode('utf-8'),row[1],row[2].encode('utf-8')])
        util.write_csv_inlist(outputfile,outputlist)

    def selectData_onekey(self,outputfile,key):
        selectStr="""SELECT * FROM paper WHERE conference like %s """
        key= self.connect.escape(key)
        
        self.cursor.execute(selectStr % (key))
        outputlist=[]
        for row in self.cursor.fetchall():
            outputlist.append([row[0].encode('utf-8'),row[1],row[2].encode('utf-8')])
        util.write_csv_inlist(outputfile,outputlist)


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

    def getYear_DBIS(self,paperDict,confDict,paper_confDict,DBIS_dict,myfilter=None):
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

    def getYear_DBIS_notmatched(self,paperDict,notmatchDict):
        outputlist=[]
        counter=0
        for key in notmatchDict:
            papername=paperDict[key]
            if papername[0]=='"' and papername[-1]=='"':
                papername=papername[1:-1]
            papername= self.connect.escape(papername)

            # print papername
            self.cursor.execute(self.selectStr_easy % (papername))
            resultlist=self.cursor.fetchall()
            if len(resultlist)>0:
                print resultlist
                year=resultlist[0][1]
                outputlist.append([key,year])
                counter+=1

        print counter
        util.write_csv_inlist('paperwithyear_dbis.csv',outputlist)
    
    def getYear_DBIS_filetered(self,paperDict,PaperRawdict,filterDict):
        outputlist=[]
        outputlist_null=[]
        counter=0
        counter_total=0

        for key in paperDict:
            if key in filterDict: continue
            papername=PaperRawdict[key]
            if papername[0]=='"' and papername[-1]=='"':
                papername=papername[1:-1]
            papername= self.connect.escape(papername)

            self.cursor.execute(self.selectStr_easy % (papername))
            resultlist=self.cursor.fetchall()
            if len(resultlist)>0:
                print resultlist
                year=resultlist[0][1]
                outputlist.append([key,year])
                counter+=1
            else:
                outputlist_null.append([key,papername])
            counter_total+=1
            print str(counter_total)+'/'+str(len(paperDict)-len(filterDict))+' '+papername

        print counter
        util.write_csv_inlist('paperwithyear_dbis.csv',outputlist)
        util.write_csv_inlist('paperwithnoresults.csv',outputlist_null)


if __name__ == "__main__":
    pod=ParseOriginalData()
    pod.readPaper('E:\\data\\dblp\\base-csv\\paper.csv')
    pod.updatePaper('E:\\data\\dblp\\mydata\\original.csv')

    paperObjDict=pickle.load(open('paperDict_obj.dat', "rb"))
    pod.matchAminerAndDBLP(paperObjDict)
    # with open('original_withvenue.csv', 'wb') as f:
    #     writer = csv.writer(f)
    #     for key in paperDict:
    #         writer.writerow([paperDict[key].id,paperDict[key].title,paperDict[key].year,paperDict[key].venue])
    # pp.run('E:\\data\\dblp\\base-csv\\paper.csv','E:\\data\\dblp\\base-csv\\person.csv','E:\\data\\dblp\\original-data\\AMiner-Coauthor.txt','E:\\data\\dblp\\base-csv\\refs.csv')
    
    # pcd=ParseCleanedData()
    # paperdict=pcd.readPaper_raw('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt')
    # confdict=pcd.readConf_raw('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_conf.txt')
    # paper_confdict=pcd.readPaper_Conf('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt')
    # dbis_dict=pcd.readDBISDict('E:\\data\\dblp\\mydata\\DBIStoCleaned_20.csv')
    
    # pcd.findDBIS_paper_withfilter(paper_confdict,paperdict,dbis_dict,myfilter)
  
    # filterdict=pcd.findDBIS_paper('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt','E:\\data\\dblp\\mydata\\DBIStoCleaned.csv')
    # pcd.matchTime(paperDict,'C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt',filterdict)
    # pcd.outputProtentialPapers(paperDict,'./notmatched.csv','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper_conf.txt','C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\paper.txt','E:\\data\\dblp\\mydata\\DBIStoCleaned.csv',)
    # pcd.processDBIS('C:\\Users\\wangzhaoyuan\\Desktop\\net_aminer\\net_aminer\\id_conf.txt','E:\\data\\dblp\\sun\\DBIS\\conf.txt')
    # pcd.getYears(sys.argv[1],str(sys.argv[2]))
    # time.sleep(10)
    # authors = dblp.search('dfahkfhnlkamefl;keakwf;la')
    # print authors.empty
    # print authors["Year"][0]==None

    # focusedDict=util.read_csv_withdict('E:\\data\\dblp\\mydata\\paperwithyear_dbis.csv',0,1)
    # pdb=ProcessDataBase()
    # notmatchDict=util.read_csv_withdict('notmatched.csv',0,1)
    # pdb.getYear_DBIS_filetered(filterdict,paperdict,focusedDict)
    # pdb.selectData_twokey('./temporal/ecml&pkdd.csv','%pkdd%','%ecml%')

    # pdb.test()
    # pdb.getYear_DBIS(paperdict,confdict,paper_confdict,dbis_dict)

    # focusedDict=util.read_csv_withdict('E:\\data\\dblp\\mydata\\paperwithyear_dbis.csv',0,1)
    # sta=Statistics(focusedDict,paper_confdict,confdict,dbis_dict)
    # sta.yearDistribution()
    # sta.output()

    # outputlist=[]
    # print len(filterdict)-len(focusedDict)
    # for key in filterdict:
    #     if key in focusedDict: continue
    #     papername=paperdict[key]
    #     outputlist.append([papername])
    # util.write_csv_inlist('./temporal/nopapers.csv',outputlist)