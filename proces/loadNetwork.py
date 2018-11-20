# -*- coding: utf-8 -*-
import os
import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
sys.path.append('../')

path='F:\\data\\dblp\\graph\\lcc-author-citation-graph-2013-2014.graphml.gz'
mygraph=nx.read_graphml(path)
nx.draw(mygraph)
nx.draw_random(mygraph)
nx.draw_circular(mygraph)
nx.draw_spectral(mygraph)

nx.draw(mygraph)
plt.savefig("path.png")
plt.show()