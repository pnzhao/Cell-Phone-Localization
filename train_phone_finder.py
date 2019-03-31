# -*- coding: utf-8 -*-
"""
This is my implementation of the phone detection task, for the intern application at Brain Corporation.
By Puning Zhao
"""

import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

directory=sys.argv[1]

'''
Parameter setting. segment_thresh is the threshold for image segmentation. 

segment_thresh: Two pixels that are spatially near to each other, and the color 
distance (color distance is defined below) is less than segment_thresh will be 
grouped into the same cluster.

bilafilter_space: the spatial kernel size used in bilateral filtering. The value
of this parameter is higher if the image is noisy.

bilafilter_range: the range kernel size used in bilateral filtering. 
'''

segment_thresh=30
bilafilter_space=50
bilafilter_range=3

#Calculate the Manhattan distance between the color of two pixels.
def color_dist(x1,y1,x2,y2,I):
    dif=I[y2][x2].astype(int)-I[y1][x1].astype(int)
    return np.sum(np.abs(dif))

#Calculate the Eucledian distance between the location of two pixels.
def coordinate_dist(p1,p2):
    return np.sqrt((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)

'''
The following code is designed for image segmentation by setting an initial seed and then let it to grow.
'''

def segment(I,thresh):
    L_segments=[]
    M=len(I)
    N=len(I[0])
    pts=set()
    for i in range(M):
        for j in range(N):
            pts.add((j,i))
    while len(pts)>0:
        p=pts.pop()
        Q={p} #temporary set for searching
        S={p}
        while Q:
            p=Q.pop()
            x=p[0]
            y=p[1]
            if x>0 and (x-1,y) in pts and color_dist(x,y,x-1,y,I)<thresh:
                Q.add((x-1,y))
                pts.remove((x-1,y))
                S.add((x-1,y))
            if x<N-1 and (x+1,y) in pts and color_dist(x,y,x+1,y,I)<thresh:
                Q.add((x+1,y))
                pts.remove((x+1,y))
                S.add((x+1,y))
            if y>0 and (x,y-1) in pts and color_dist(x,y,x,y-1,I)<thresh:
                Q.add((x,y-1))
                pts.remove((x,y-1))
                S.add((x,y-1))
            if y<M-1 and (x,y+1) in pts and color_dist(x,y,x,y+1,I)<thresh:
                Q.add((x,y+1))
                pts.remove((x,y+1))
                S.add((x,y+1))
        if len(S)>100 and len(S)<1000:
            L_segments.append(S)
    return L_segments  

def checkrectangle(segment):
    s=list(segment)
    pleft=s[0]
    pright=s[0]
    ptop=s[0]
    pbottom=s[0]
    for p in s:
        if p[0]<pleft[0]:
            pleft=p
        elif p[0]>pright[0]:
            pright=p
        elif p[1]<ptop[1]:
            ptop=p
        elif p[1]>pbottom[1]:
            pbottom=p
    pnew=(ptop[0]+pbottom[0]-pleft[0],ptop[1]+pbottom[1]-pleft[1])
    d=coordinate_dist(pnew,pright)
    plefttop=s[0]
    prighttop=s[0]
    prightbottom=s[0]
    pleftbottom=s[0]
    for p in s:
        if p[0]+p[1]<plefttop[0]+plefttop[1]:
            plefttop=p
        elif p[0]-p[1]<prighttop[0]-prighttop[1]:
            prighttop=p
        elif p[0]-p[1]>pleftbottom[0]-pleftbottom[1]:
            pleftbottom=p
        elif p[0]+p[1]>prightbottom[0]+prightbottom[1]:
            prightbottom=p
    pnew=(prighttop[0]+pleftbottom[0]-plefttop[0],prighttop[1]+pleftbottom[1]-plefttop[1])
    d=min(d,coordinate_dist(pnew,prightbottom))
    if d<6:
        return True
    else:
        return False

def circumference(segment):
    c=0
    for p in segment:
        if (p[0]-1,p[1]) not in segment:
            c+=1
        elif (p[0]+1,p[1]) not in segment:
            c+=1
        elif (p[0],p[1]-1) not in segment:
            c+=1
        elif (p[0],p[1]+1) not in segment:
            c+=1
    return c**2/len(segment)

def localize(I,segment_thresh):
    Iblur=cv2.bilateralFilter(I,50,bilafilter_range,bilafilter_space)
    I_segments=segment(Iblur,segment_thresh)
    M=len(Iblur)
    N=len(Iblur[0])
    if not I_segments:
        return (-1,-1)
    else:
        xc=-1
        yc=-1
        mingray=255
        Igray=cv2.cvtColor(Iblur,cv2.COLOR_BGR2GRAY)
        S1=[]
        S2=[]
        S3=[]
        for seg in I_segments:
            if circumference(seg)<25 and checkrectangle(seg):
                S1.append(seg)
            elif circumference(seg)<25:
                S2.append(seg)
            else:
                S3.append(seg)
        if S1:
            S=S1
        elif S2:
            S=S2
        else:
            S=S3
        for seg in S:
            gray=np.mean([Igray[p[1]][p[0]] for p in seg])
            if gray<mingray:
                mingray=gray
                xc=np.mean([p[0] for p in seg])/N
                yc=np.mean([p[1] for p in seg])/M
        return (xc,yc)
    
def accuracyeval(xpredict,ypredict,xlocs,ylocs):
    N_figs=len(xpredict)
    correct=0
    miss=set()
    for i in range(N_figs):
        if np.sqrt((xpredict[i]-xlocs[i])**2+(ypredict[i]-ylocs[i])**2)<0.05:
            correct+=1
        else:
            miss.add(indices_sorted[i])
    print(correct/N_figs)
    print('The following images are wrong localized:')
    print(miss)
    
#Read the data file.
d=pd.read_csv(directory+'\labels.txt',delimiter=' ',header=None)
N_figs=len(d[0])
indices=np.array([0]*N_figs)
for i in range(N_figs):
    indices[i]=int(d[0][i].split('.')[0])
xlocs=[a for _,a in sorted(zip(indices,d[1]))]
ylocs=[a for _,a in sorted(zip(indices,d[2]))]
dnew=pd.DataFrame({'0':sorted(indices),'1':xlocs,'2':ylocs})
indices_sorted=sorted(indices)

#Localize each train image.
x_estimate=np.array([0.]*N_figs)
y_estimate=np.array([0.]*N_figs)
for k in range(N_figs):
    ind=indices_sorted[k]
    print(f'Image {ind}:')
    filename=os.path.join(directory, str(ind) + '.jpg')
    I=plt.imread(filename)
    loc=localize(I,segment_thresh)
    x_estimate[k]=loc[0]
    y_estimate[k]=loc[1]
    print(f'Estimated location: x={loc[0]:.4f}, y={loc[1]:.4f}. Real location: x={xlocs[k]:.4f}, y={ylocs[k]:.4f}.')

print('Accuracy:')
accuracyeval(x_estimate,y_estimate,xlocs,ylocs)
