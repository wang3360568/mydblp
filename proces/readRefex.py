# -*- coding: utf-8 -*-
import os
import json
import sys
import collections
import util
import re
import csv
import time
import HTMLParser
import cPickle as pickle
sys.path.append('.')
from myclass.myobj import Paper,Person,Author,Person_node
import snap
import math
import collections
import numpy as np
import pandas as pd

class ReadRefex():
    
    def __init__(self,fileFolder,start,end,inOrderList):
        self.fileFolder=fileFolder
        self.start=start
        self.end=end
        self.inOrderList=inOrderList
    
    def readfile(self):
        refexFeatures=collections.OrderedDict()
        for year in range(self.start,self.end+1):
            contentDict=dict()
            fp=open(self.fileFolder+str(year)+'-featureValues.csv','r')
            while 1:
                line = fp.readline()
                if not line:
                    break
                content=line.strip().split(',')
                numericContent= map(float, content[1:])
                contentDict[content[0]]=numericContent
            keylist=self.inOrderList[year]
            outputlist=[]
            for key in keylist:
                outputlist.append(contentDict[key])
            farray=np.array(outputlist)
            refexFeatures[year]=farray
        return refexFeatures

