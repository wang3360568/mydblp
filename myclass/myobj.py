# -*- coding: utf-8 -*-
import os
import json
import sys

class Person:
    def __init__(self,id,name):
        self.id=id
        self.name=name
        self.papers=dict()
    def update_paper(self,paper_id,rank):
        self.papers[paper_id]=rank

class Author:
    def __init__(self,authorid,paperid):
        self.authorid=int(authorid)
        self.paperid=int(paperid)

class Paper:
    def __init__(self, id, title, venue, year,abstract=None):

        try:
            self.id = id
            self.title = title
            self.venue = venue
            self.year = int(year) if year else None
            self.abstract = abstract if abstract else None
            self.author_ids = {}
            self.referring_ids = []
            self.referred_ids = []
        except:
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

