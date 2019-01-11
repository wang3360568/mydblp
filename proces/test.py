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
# sys.path.append('../snap_jure')
import snap
import math
import collections
import numpy as np




Network = snap.GenRndGnm(snap.PNEANet, 10, 50)
labels = snap.TIntStrH()
for NI in Network.Nodes():
    labels[NI.GetId()] = str(NI.GetId())
snap.DrawGViz(Network, snap.gvlDot, "output.png", " ", labels)