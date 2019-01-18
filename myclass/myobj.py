# -*- coding: utf-8 -*-
import os
import json
import sys
import copy
import collections

class Person:
    def __init__(self,id,name):
        self.id=id
        self.name=name
        self.papers=dict()
        self.yearDict=collections.OrderedDict()
        self.areaDict=collections.OrderedDict()
    def update_paper(self,paper_id,rank):
        self.papers[paper_id]=rank
    
    def update_byyear(self,startYear,endYear,paperDict):
        for i in range(startYear,endYear):
            self.yearDict[i]=[]
        for key in self.papers:
            if paperDict[key].year<=startYear:
                self.yearDict[startYear].append(key)
            else:
                self.yearDict[paperDict[key].year].append(key)

    def countArea(self,startYear,endYear,paperDict):
        for i in range(startYear,endYear):
            self.areaDict[i]={'DM':0,'IR':0,'ML':0,'DB':0}
        for key in self.papers:
            if paperDict[key].year>endYear: continue
            theArea=paperDict[key].area
            theYear=paperDict[key].year
            if theYear<=startYear:
                for i in range(startYear,endYear):
                    self.areaDict[i][theArea]=self.areaDict[i][theArea]+1
            else:
                for i in range(theYear,endYear):
                    self.areaDict[i][theArea]=self.areaDict[i][theArea]+1

class Person_node:
    def __init__(self,authorId,learningYear):
        self.authorId=authorId
        self.learningYear=learningYear
        self.hisval=0.0
        self.val=0.0

    def updateHisVal(self,val):
        self.hisval=val

    def updateVal(self,val):
        self.val=val

    def __eq__( self, other ):
        return self.authorId == other.authorId and self.learningYear == other.learningYear

    def __hash__(self):
         return hash(self.authorId) ^ hash(self.learningYear)


class Author:
    def __init__(self,authorid,paperid):
        self.authorid=int(authorid)
        self.paperid=int(paperid)

class Paper:
    def __init__(self, id, title, venue, year,area,authors=None, abstract=None):

        try:
            self.id = id
            self.title = title
            self.venue = venue
            self.year = int(year) if year else None
            self.abstract = abstract if abstract else None
            if authors:
                self.authors=copy.copy(authors)
            else:
                self.authors=[]
            self.author_ids = {}
            self.referring_ids = []
            self.referred_ids = []
            self.area=area
        except:
            print 'error'
            print id
            print title
            print venue
            print year 

    def update_authorId(self,author_id,rank):
        self.author_ids[rank]=author_id
    
    def update_referringIds(self,paper_id):
        self.referring_ids.append(paper_id)

    def update_referredIds(self,paper_id):
        self.referred_ids.append(paper_id)

class Ref:
    def __init__(self,paperid,refid):
        self.refid=int(refid)
        self.paperid=int(paperid)

