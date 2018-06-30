# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def closest_pt(p , pts):
    global mydist
    #pts[cdist([p], pts).argmin] 
    mydist = cdist([p], pts)
    closest_index = cdist([p], pts).argmin()
    return mydist
    
    
def sort_pt(pts):
    global sortList
    sortList = sorted(pts, key = lambda e: cdist([e],[pts[0]]))  
    return sortList


from sklearn.neighbors import NearestNeighbors
import networkx as nx


def sort_pt_new(pts):
    global sortList, G, x, y ,xx ,yy, order , x1,y1, paths
    x, y = zip(*[(a[0] , a[1]) for a in pts])
    x = np.array(x)
    y= np.array(y)
    pts = np.c_[x,y]

    clf = NearestNeighbors(2).fit(pts)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    
    order = list(nx.dfs_preorder_nodes(T,0))
    xx = x[order]
    yy = y[order]

    
#    paths = [list(nx.dfs_preorder_nodes(T,i)) for i in range(len(pts))]
#    
#    mindist = np.inf
#    minidx = 0
#    for i in range(len(pts)):
#        p = paths[i]
#        ordered = pts[p]
#        cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
#        if cost < mindist:
#            mindist = cost
#            minidx = i
#            
#    opt_order = paths[minidx]
#    xx = x[opt_order]
#    yy = y[opt_order]
#    plt.plot(xx,yy)
    
    sortList = sorted(pts, key = lambda e: cdist([e],[pts[0]]))  
    return sortList


   
#pnts = ((np.loadtxt('sandMap.txt')).astype(int)).tolist()
#pnts = np.array(pnts).reshape(int(len(pnts)/2), 2).tolist()
   # print("List Sorted" )

#print(sort_pt_new(pnts))

