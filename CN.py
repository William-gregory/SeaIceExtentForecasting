import numpy as np
import warnings
import datetime
import itertools
import os
import math
import operator
import random
from scipy import stats
import glob

class Network:
    def __init__(self, dimX=0, dimY=0, nodes=[], corrs=[], tau=0, V={}, unavail=[], anomaly={}, links={}):
        self.dimX = dimX
        self.dimY = dimY
        self.nodes = nodes
        self.corrs = corrs
        self.tau = tau
        self.V = V
        self.unavail = unavail
        self.anomaly = anomaly
        self.links = links
    
    def cell_level(self, data, tag, datapath):
        print('Creating cell-level network')
        print(datetime.datetime.now())
        t = data.shape[2]
        files = range(self.dimX*self.dimY)
        nodes = self.dimX*self.dimY
        count_array = []
        if not os.path.exists(datapath+'/correlations_'+str(tag)):
            os.makedirs(datapath+'/correlations_'+str(tag))
            count = -1
            latest = 0
        elif (os.path.exists(datapath+'/correlations_'+str(tag))==True) & (os.path.exists(datapath+'/correlations_'+str(tag)+'/nodes.txt')==False):
            pwdir = os.getcwd()
            os.chdir(datapath+'/correlations_'+str(tag))
            fss = glob.glob('*')
            if len(fss) == 0:
                count = -1
                latest = 0
            else:
                latest = max(fss, key=os.path.getctime)
                count = int(latest)
                latest = int(latest) + 1
                os.chdir(pwdir)
                count_array = [int(f) for f in fss]
        elif (os.path.exists(datapath+'/correlations_'+str(tag))==True) & (os.path.exists(datapath+'/correlations_'+str(tag)+'/nodes.txt')==True):
            print('Correlations already pre-computed. Will load them instead')
            return 
        
        for node1 in range(latest,nodes):
            ix = int(math.floor(node1/self.dimY))
            jx = int(node1 % self.dimY)
            if (~np.isnan(data[ix,jx,range(t)]).all()) & (max(abs(data[ix,jx,range(t)])) > 0):
                count += 1
                count_array.append(count)
                R = np.empty((self.dimX,self.dimY))
                for node2 in range(nodes):
                    iy = int(math.floor(node2/self.dimY))
                    jy = int(node2 % self.dimY)
                    if (node1 != node2) & (~np.isnan(data[iy,jy,range(t)]).all()) & (max(abs(data[iy,jy,range(t)])) > 0):
                        R[iy,jy] = stats.pearsonr(data[ix,jx,range(t)],data[iy,jy,range(t)])[0]
                    else:
                        R[iy,jy] = np.nan
                self.corrs.append(R)
                np.savetxt(datapath+'/correlations_'+str(tag)+'/'+str("%02d"%files[count]), R, delimiter='\t')
            else:
                count += 1
                
        np.savetxt(datapath+'/correlations_'+str(tag)+'/nodes.txt',count_array,fmt='%i',newline='\n')
        self.corrs = np.array(self.corrs)

        print('Done!')
    
    def tau(self, data, alpha, tag, datapath):
        print('Generating threshold factor')
        print(datetime.datetime.now())
        
        self.nodes = np.genfromtxt(datapath+'/correlations_'+str(tag)+'/nodes.txt')
        
        if len(self.corrs) == 0:
            self.corrs = np.zeros((self.nodes.shape[0],self.dimX,self.dimY))
            for n in range(self.nodes.shape[0]):
                self.corrs[n,:,:] = np.genfromtxt(datapath+'/correlations_'+str(tag)+'/'+str("%02d"%self.nodes[n]), autostrip=True)
        else:
            pass
        
        df = data.shape[2] - 2
        R = np.copy(self.corrs).ravel()
        R[R<0] = np.nan
        R = R[~np.isnan(R)]
        T = R*np.sqrt(df/(1 - R**2))
        P = stats.t.sf(T,df)
        R = R[P<alpha]

        self.tau = np.nanmean(R)
    
        print('Network threshold factor =', '%.3f' % self.tau)
    
    def area_level(self, data, latlon_grid=False):
        ids = np.where(np.isnan(data[:,:,:]))
        i_nan = ids[0][0] ; j_nan = ids[1][0]

        def cell_neighbours(i, j, i_nan, j_nan):
            if [i-1,j] not in self.unavail:
                nei_1 = [i-1,j] if 0 <= j <= self.dimY-1 and 0 <= i-1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_1 = [i_nan,j_nan]
            if [i+1,j] not in self.unavail:
                nei_2 = [i+1,j] if 0 <= j <= self.dimY-1 and 0 <= i+1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_2 = [i_nan,j_nan]
            if ([i,j-1] not in self.unavail) & (latlon_grid==False):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j-1] not in self.unavail) & (latlon_grid==True):
                nei_3 = [i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i,self.dimY-1]
            elif [i,j-1] in self.unavail:
                nei_3 = [i_nan,j_nan]
            if ([i,j+1] not in self.unavail) & (latlon_grid==False):
                nei_4 = [i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i_nan,j_nan]
            elif ([i,j+1] not in self.unavail) & (latlon_grid==True):
                nei_4 = [i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <= i <= self.dimX-1 else [i,0]
            elif [i,j+1] in self.unavail:
                nei_4 = [i_nan,j_nan]
            return [nei_1, nei_2, nei_3, nei_4]

        def area_neighbours(Area, i_nan, j_nan):
            neighbours = []
            for cell in Area:
                if [cell[0]-1,cell[1]] not in self.unavail:
                    neighbours.append([cell[0]-1,cell[1]] if 0 <= cell[1] <= self.dimY-1 and 0 <= cell[0]-1 <= self.dimX-1 else [i_nan,j_nan])
                else:
                    neighbours.append([i_nan,j_nan])
                if [cell[0]+1,cell[1]] not in self.unavail:
                    neighbours.append([cell[0]+1,cell[1]] if 0 <= cell[1] <= self.dimY-1 and 0 <= cell[0]+1 <= self.dimX-1 else [i_nan,j_nan])
                else:
                    neighbours.append([i_nan,j_nan])
                if ([cell[0],cell[1]-1] not in self.unavail) & (latlon_grid==False):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]-1] not in self.unavail) & (latlon_grid==True):
                    neighbours.append([cell[0],cell[1]-1] if 0 <= cell[1]-1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i,self.dimY-1])
                elif [cell[0],cell[1]-1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
                if ([cell[0],cell[1]+1] not in self.unavail) & (latlon_grid==False):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i_nan,j_nan])
                elif ([cell[0],cell[1]+1] not in self.unavail) & (latlon_grid==True):
                    neighbours.append([cell[0],cell[1]+1] if 0 <= cell[1]+1 <= self.dimY-1 and 0 <= cell[0] <= self.dimX-1 else [i,0])
                elif [cell[0],cell[1]+1] in self.unavail:
                    neighbours.append([i_nan,j_nan])
            return neighbours

        def area_max_correlation(Area, neighbours):
            Rmean = [] ; X = []
            for cell in neighbours:
                R = []
                new_node = cell[0]*self.dimY + cell[1]
                if new_node in self.nodes:
                    X.append(cell)
                    IDnew = np.where(self.nodes == new_node)
                    IDnew = int(IDnew[0])
                    for cells in Area:
                        if ([cells[0],cells[1]] != [cell[0],cell[1]]):
                            R.append(self.corrs[IDnew,cells[0],cells[1]])
                    Rmean.append(np.nanmean(R))
            try:
                Rmax = np.nanmax(Rmean)
            except ValueError:
                Rmax = np.nan
            return X, Rmean, Rmax

        #S T E P   1   (C R E A T E   A R E A S)

        self.V = {}
        self.unavail = []
        k = 0
        print('Creating area-level network')  
        print(datetime.datetime.now())
        for i,j in itertools.product(range(self.dimX),range(self.dimY)):
            node = i*self.dimY + j
            if node in self.nodes:
                ID = np.where(self.nodes == node)
                ID = int(ID[0])
                if [i,j] not in self.unavail:
                    while True:
                        neighbours = cell_neighbours(i, j, i_nan, j_nan)
                        neighbour_corrs = [self.corrs[ID,neighbours[0][0],neighbours[0][1]],
                                           self.corrs[ID,neighbours[1][0],neighbours[1][1]],
                                           self.corrs[ID,neighbours[2][0],neighbours[2][1]],
                                           self.corrs[ID,neighbours[3][0],neighbours[3][1]]]
                        maxR = np.nanmax(neighbour_corrs)
                        if maxR > self.tau:
                            maxID = np.where(neighbour_corrs==maxR)
                            if np.shape(maxID) == 1:
                                maxID = int(maxID[0])
                                maxID = neighbours[maxID]
                            else:
                                maxID = int(maxID[0][0])
                                maxID = neighbours[maxID]
                            if ([i,j] not in self.unavail) and ([maxID[0],maxID[1]] not in self.unavail):
                                self.V.setdefault(k, []).append([i,j])
                                self.V.setdefault(k, []).append([maxID[0],maxID[1]])
                                self.unavail.append([i,j])
                                self.unavail.append([maxID[0],maxID[1]])

                                while True: #expand
                                    neighbours = area_neighbours(self.V[k], i_nan, j_nan)
                                    X, Rmean, Rmax = area_max_correlation(Area=self.V[k], neighbours=neighbours)
                                    if Rmax > self.tau:
                                        RmaxID = np.where(Rmean==Rmax)
                                        if np.shape(RmaxID) == 1:
                                            RmaxID = int(RmaxID[0])
                                            m = X[RmaxID]
                                        else:
                                            RmaxID = int(RmaxID[0][0])
                                            m = X[RmaxID]
                                        if m not in self.unavail:
                                            self.V.setdefault(k, []).append([m[0],m[1]])
                                            self.unavail.append([m[0],m[1]])
                                        else:
                                            break
                                    else:
                                        break
                                k = k + 1
                            else:
                                break
                        else:
                            break

        self.unavail = []
        while True:
            Rs = {}
            unavail_neighbours = {}
            num_cells = dict([(area,len(self.V[area])) if self.V[area] not in self.unavail else (area,0) for area in self.V.keys()])
            maxID = max(num_cells.items(), key=operator.itemgetter(1))[0]
            if num_cells[maxID] == 0:
                break
            else:
                neighbours = area_neighbours(self.V[maxID], i_nan, j_nan)
                for cell in neighbours:
                    node = cell[0]*self.dimY + cell[1]
                    Rmean = []                   
                    if (node in self.nodes) & (cell not in self.V[maxID]) & (cell not in [k for k, g in itertools.groupby(sorted(itertools.chain(*unavail_neighbours.values())))]) & (len([area for area, cells in self.V.items() if cell in cells]) > 0):
                        nID = [area for area, cells in self.V.items() if cell in cells][0]
                        unavail_neighbours[nID] = self.V[nID]
                        X, Rmean, Rmax = area_max_correlation(Area=self.V[nID]+self.V[maxID], neighbours=self.V[nID]+self.V[maxID])
                        if nID not in Rs: 
                            Rs[nID] = np.nanmean(Rmean)
                try:
                    Rs_maxID = max(Rs.items(), key=operator.itemgetter(1))[0]
                    if Rs[Rs_maxID] > self.tau:
                        for cell in self.V.pop(Rs_maxID, None):
                            self.V.setdefault(maxID, []).append([cell[0],cell[1]])
                    else:
                        self.unavail.append(self.V[maxID])
                except ValueError:
                    self.unavail.append(self.V[maxID])

        print('Done!')
                
    def intra_links(self, data, area=None, lat=None, cellsize=None):
        print('Generating network links')
        print(datetime.datetime.now())
        self.anomaly = {}
        self.links = {}
        if lat is not None:
            scale = np.sqrt(np.cos(np.radians(lat)))
        elif area is not None:
            scale = np.sqrt(area)
        elif cellsize is not None:
            scale = np.ones((data.shape[0],data.shape[1]))*np.sqrt(cellsize)
        else:
            scale = np.ones((data.shape[0],data.shape[1]))
        for A in self.V:
            temp_array = np.zeros((data.shape))
            temp_array[temp_array==0] = np.nan
            for cell in self.V[A]:
                temp_array[cell[0],cell[1],:] = np.multiply(data[cell[0],cell[1],:],scale[cell[0],cell[1]])
            temp = np.nansum(temp_array, axis=(0,1))
            self.anomaly[A] = temp
          
        for A in self.anomaly: 
            for A2 in self.anomaly:
                if A2 != A:
                    self.links.setdefault(A, []).append(np.cov(self.anomaly[A],self.anomaly[A2], bias=True))
                elif A2 == A:
                    self.links.setdefault(A, []).append(np.nan)
            
        #for A in self.links:
        #    absolute = []  
        #    for i in self.links[A]:
        #        if ~np.isnan(i):
        #            absolute.append(abs(i))
        #    self.strength[A] = np.nansum(absolute)
        
