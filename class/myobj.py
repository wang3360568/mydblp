# -*- coding: utf-8 -*-
import os
import json
import sys
sys.path.append('../')

class Person:
    def __init__(self,id,name):
        self.id=int(id)
        self.name=name


class Paper:

    def __init__(self, id, title, authors, venue, refs, abstract, year):
        self.id = int(id)
        self.title = title
        self.venue = venue
        self.refs = [int(ref) for ref in refs]
        self.year = int(year) if year else None
        self.authors = [a for a in authors.split(',') if a]
        self.abstract = abstract if abstract else None


class Author:

    def __init__(self,authorid,paperid):
        self.authorid=int(authorid)
        self.paperid=int(paperid)


class Ref:

    def __init__(self,paperid,refid):
        self.refid=int(refid)
        self.paperid=int(paperid)

