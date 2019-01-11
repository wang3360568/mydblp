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
sys.path.append('./')
from myclass.myobj import Paper,Person,Author,Person_node
# sys.path.append('../snap_jure')
from setupNetwork import SetupNetwork
from readRefex import ReadRefex
import math
import collections
import numpy as np


class InitNetworks:

    startYear_total=1998
    endYear=2014
    startYear_learning=2005
    gap=2
    hisGap=10
    rou=0.4
    lowerVal=0.001 #0.005

    def __init__(self,paperObjFile,authorObjFile,yearCountFile):
        self.paperObjDict=pickle.load(open(paperObjFile, "rb"))
        self.authorObjDict=pickle.load(open(authorObjFile, "rb"))
        self.yearCountDict=pickle.load(open(yearCountFile, "rb"))
        self.author_nodes=collections.OrderedDict()

    def selectAuthors_forARange(self,learningYear):
        # total=lambda :total+self.yearCountDict[learningYear+i] for i in range(gap+1)
        nodeDict=dict()
        for key in self.authorObjDict:
            author=self.authorObjDict[key]
            thisYearDict=author.yearDict
            val=0.0
            hisVal=0.0
            for yearkey in thisYearDict:
                if yearkey<learningYear-self.gap and yearkey>=learningYear-self.gap-10:
                    hisVal=hisVal+ (float(len(thisYearDict[yearkey]))/self.yearCountDict[yearkey])*math.exp(-self.rou*(learningYear-self.gap-yearkey))
                if yearkey>=learningYear-self.gap and yearkey<=learningYear:
                    val+= float(len(thisYearDict[yearkey]))/self.yearCountDict[yearkey]
            if val==0.0 and hisVal==0.0: continue
            author_node=Person_node(authorId=key,learningYear=learningYear)
            author_node.updateHisVal(hisVal)
            author_node.updateVal(val)
            nodeDict[key]=author_node
        return nodeDict

    def selectAuthors(self):
        for i in range(self.startYear_learning,self.endYear):
            nodeDict=self.selectAuthors_forARange(i)
            pre_nodeDict=None
            newDict=dict()
            if i>self.startYear_learning:
                # pre_nodeDict=self.author_nodes[i-1]
                pre_nodeDict=None
            for key in nodeDict:
                if pre_nodeDict:
                    if key in pre_nodeDict:
                        newDict[key]=nodeDict[key]
                    else:
                        author=nodeDict[key]
                        if author.val+author.hisval>=self.lowerVal:
                            newDict[key]=author
                else:
                    author=nodeDict[key]
                    if author.val+author.hisval>=self.lowerVal:
                        newDict[key]=author
            self.author_nodes[i]=newDict
        return self.author_nodes
    
    def generateEdge(self):
        edgeDict=collections.OrderedDict()
        edgeIndexDict=collections.OrderedDict()
        for yearkey in self.author_nodes:
            author_node=self.author_nodes[yearkey]
            key_list=[]
            for key in author_node:
                key_list.append(key)
            mlength=len(key_list)
            ajmatrix=np.zeros((mlength,mlength))

            for i in range(mlength):
                ikey=key_list[i]
                iauthor=self.authorObjDict[ikey]
                for pkey in iauthor.papers:
                    thePaper= self.paperObjDict[pkey]
                    theYear=thePaper.year
                    if theYear>yearkey: continue
                    theAuthors=thePaper.authors
                    for akey in theAuthors:
                        if akey==ikey: continue
                        if akey in key_list:
                            akey_index=key_list.index(akey)
                            val=math.exp(-self.rou*(yearkey-theYear))
                            ajmatrix[i][akey_index]=ajmatrix[i][akey_index]+val
                            # ajmatrix[akey_index][i]=ajmatrix[i][akey_index]
            # b = np.nonzero(ajmatrix)
            # print(np.array(b).ndim)
            np.savetxt('./proces/temporal/year_'+str(yearkey)+'.csv',ajmatrix,fmt='%d',delimiter=',')
            util.write_csv_inlist('./proces/temporal/nodeslist_'+str(yearkey)+'.csv',key_list)
            edgeDict[yearkey]=ajmatrix
            edgeIndexDict[yearkey]=key_list
        return edgeDict,edgeIndexDict

if __name__ == "__main__":
    init=InitNetworks('./proces/paperDict_obj.dat','./proces/authorDict_obj.dat','./proces/yearCount.dat')
    nodedict=init.selectAuthors()

    # for yearkey in nodedict:
    #     author_nodes=nodedict[yearkey]
    #     outputlist=[]
    #     for key in author_nodes:
    #         author=author_nodes[key]
    #         outputlist.append([key,author.val,author.hisval,author.val+author.hisval])

    #     util.write_csv_inlist('./proces/temporal/nodes_'+str(yearkey)+'.csv',outputlist)

    edgeDict,nodeIndexDict=init.generateEdge()

    sN=SetupNetwork(edgeDict,nodeIndexDict,init.paperObjDict)
    edgeDict_new,nodeIndexDict_new=sN.buildNetwork()
    attributeslist=sN.getNodeAttributes()

    # refex=ReadRefex('E:\\data\\dblp\\mydata\\refexfeatures_'+str(lowerVal)+'\\',2005,2013,nodeIndexDict_new)
    # refexFeature=refex.readfile()

    pickle.dump(edgeDict_new, open('edgeDict_'+str(init.lowerVal)+'.dat', "wb"), True)
    pickle.dump(nodeIndexDict_new, open('nodeIndexDict_'+str(init.lowerVal)+'.dat', "wb"), True)
    pickle.dump(attributeslist, open('attributeslist_'+str(init.lowerVal)+'.dat', "wb"), True)
    # pickle.dump(refexFeature, open('refexFeature_'+str(init.lowerVal)+'.dat', "wb"), True)

    # for i in range(1999,2016):
    #     author_nodes=init.selectAuthors_forARange(i)
    #     outputlist=[]
    #     for key in author_nodes:
    #         author=author_nodes[key]
    #         outputlist.append([key,author.val,author.hisval,author.val+author.hisval])

    #     util.write_csv_inlist('./proces/temporal/nodes_'+str(i)+'.csv',outputlist)

    # mtne=MTNE_nocompany(edgeDict,nodeIndexDict)
    # u_list,a_list,F=mtne.MTNE()
    # pickle.dump(u_list, open('u_list.dat', "wb"), True)
    # pickle.dump(a_list, open('a_list.dat', "wb"), True)
    # pickle.dump(F, open('F.dat', "wb"), True)

    # mtne=MTNE_learnSimilarity_nocompany(edgeDict,nodeIndexDict)
    # a_list,f_list,s=mtne.MTNE()
    # pickle.dump(a_list, open('A_list.dat', "wb"), True)
    # pickle.dump(f_list, open('F_list.dat', "wb"), True)
    # pickle.dump(s, open('S.dat', "wb"), True)
