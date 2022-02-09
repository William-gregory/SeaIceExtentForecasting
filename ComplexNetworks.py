import numpy as np
import warnings
import datetime
import itertools
import os
import math
import operator
import random
from scipy import stats

class Network:
    def __init__(self,data,V={},A={},corrs=[],tau=0,nodes=[],unavail=[],anomaly={},links={},strength={},strengthmap=[]):
        """
        The input 'data' are expected to be de-trended (zero-mean)
        and in the format x,y,t if an area grid, or lat,lon,t for
        a lat-lon grid.
        """
        self.data = data
        self.dimX,self.dimY,self.dimT = self.data.shape
        self.V = V
        self.A = A
        self.corrs = corrs
        self.tau = tau
        self.nodes = nodes
        self.unavail = unavail
        self.anomaly = anomaly
        self.links = links
        self.strength = strength
        self.strengthmap = strengthmap
    
    def tau(self, significance=0.01):
        ID = np.where(np.abs(np.nanmax(self.data,2))>0)
        N = np.shape(ID)[1]
        R = np.corrcoef(self.data[ID])
        np.fill_diagonal(R,np.nan)
        self.corrs = np.zeros((N,self.dimX,self.dimY))*np.nan
        self.nodes = np.atleast_2d(ID[0]*self.dimY + ID[1])
        for n in range(N):
            self.corrs[n,:,:][ID] = R[n,:]
        
        df = self.dimT - 2
        R = R[R>=0]
        T = R*np.sqrt(df/(1 - R**2))
        P = stats.t.sf(T,df)
        R = R[P<significance]

        self.tau = np.mean(R)
        
    def area_level(self, latlon_grid=False):
        ids = np.where(np.isnan(self.data))
        i_nan = ids[0][0] ; j_nan = ids[1][0]
        
        def gen_cell_neighbours(i, j, i_nan, j_nan):
            if [i-1,  j] not in self.unavail:
                nei_1 = [i-1,  j] if 0 <=   j <= self.dimY-1 and 0 <= i-1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_1 = [i_nan,j_nan]
            if [i+1,  j] not in self.unavail:
                nei_2 = [i+1,  j] if 0 <=   j <= self.dimY-1 and 0 <= i+1 <= self.dimX-1 else [i_nan,j_nan]
            else:
                nei_2 = [i_nan,j_nan]
            if ([  i,j-1] not in self.unavail) & (latlon_grid==False):
                nei_3 = [  i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <=   i <= self.dimX-1 else [i_nan,j_nan]
            elif ([  i,j-1] in self.unavail) & (latlon_grid==False):
                nei_3 = [i_nan,j_nan]
            elif ([  i,j-1] not in self.unavail) & (latlon_grid==True):
                nei_3 = [  i,j-1] if 0 <= j-1 <= self.dimY-1 and 0 <=   i <= self.dimX-1 else [i,self.dimY-1]
            elif ([  i,j-1] in self.unavail) & (latlon_grid==True):
                nei_3 = [i_nan,j_nan]
            if ([  i,j+1] not in self.unavail) & (latlon_grid==False):
                nei_4 = [  i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <=   i <= self.dimX-1 else [i_nan,j_nan]
            elif ([  i,j+1] in self.unavail) & (latlon_grid==False):
                nei_4 = [i_nan,j_nan]
            elif ([  i,j+1] not in self.unavail) & (latlon_grid==True):
                nei_4 = [  i,j+1] if 0 <= j+1 <= self.dimY-1 and 0 <=   i <= self.dimX-1 else [i,0]
            elif ([  i,j+1] in self.unavail) & (latlon_grid==True):
                nei_4 = [i_nan,j_nan]
            return nei_1, nei_2, nei_3, nei_4
        
        def gen_area_neighbours(Area, i_nan, j_nan):
            Anei_1 = []
            Anei_2 = []
            Anei_3 = []
            Anei_4 = []
            for A in range(np.shape(Area[k])[0]):
                if [Area[k][A][0]-1,Area[k][A][1]] not in self.unavail:
                    Anei_1.append([Area[k][A][0]-1,Area[k][A][1]] if 0 <=   Area[k][A][1] <= self.dimY-1 and 0 <= Area[k][A][0]-1 <= self.dimX-1 else [i_nan,j_nan])
                if [Area[k][A][0]+1,Area[k][A][1]] not in self.unavail:
                    Anei_2.append([Area[k][A][0]+1,Area[k][A][1]] if 0 <=   Area[k][A][1] <= self.dimY-1 and 0 <= Area[k][A][0]+1 <= self.dimX-1 else [i_nan,j_nan])
                if [Area[k][A][0],Area[k][A][1]-1] not in self.unavail:
                    Anei_3.append([Area[k][A][0],Area[k][A][1]-1] if 0 <=   Area[k][A][1]-1 <= self.dimY-1 and 0 <= Area[k][A][0] <= self.dimX-1 else [i_nan,j_nan])
                if [Area[k][A][0],Area[k][A][1]+1] not in self.unavail:
                    Anei_4.append([Area[k][A][0],Area[k][A][1]+1] if 0 <=   Area[k][A][1]+1 <= self.dimY-1 and 0 <= Area[k][A][0] <= self.dimX-1 else [i_nan,j_nan])
            return Anei_1, Anei_2, Anei_3, Anei_4

        def area_max_correlation(area_neighbours, Area):
            R_mean = []
            X = []
            for nei in area_neighbours:
                R = []
                if ((nei[0][0]*self.dimY)+nei[0][1]) in self.nodes[0,:]:
                    #print('Anei = ',nei)
                    X.append(nei)
                    ID_new = np.where(self.nodes[0,:] == ((nei[0][0]*self.dimY)+nei[0][1]))
                    ID_new = int(ID_new[0])
                    #print('ID_new = ',ID_new)
                    for a in range(np.shape(Area[k])[0]):
                        b = int(Area[k][a][0])
                        c = int(Area[k][a][1])
                        R.append(self.corrs[ID_new,b,c])
                    R_mean.append(np.nanmean(R))
                    #print('R_mean = ',R_mean)
            try:
                Rmax = np.nanmax(R_mean)
            except ValueError:
                Rmax = np.nan
            #print('Rmax = ',Rmax)
            return X, R_mean, Rmax

        def expand(Area):
            while True:
                Anei_1, Anei_2, Anei_3, Anei_4 = gen_area_neighbours(Area, i_nan, j_nan)
                Anei_list = [Anei_1, Anei_2, Anei_3, Anei_4]
                Anei_flat = []
                for sublist in Anei_list:
                    for item in sublist:
                        if item not in Anei_flat:
                            Anei_flat.append([item])
                #print('Anei_flat = ',Anei_flat)
                if np.shape(Anei_flat)[0] == 0:
                    return Area
                    break
                elif np.shape(Anei_flat)[0] != 0:
                    X, R_mean, Rmax = area_max_correlation(Anei_flat, Area)
                    if Rmax > self.tau:
                        Rmax_ID = np.where(R_mean==Rmax)
                        if np.shape(Rmax_ID) == 1:
                            Rmax_ID = int(Rmax_ID[0])
                            m = X[Rmax_ID]
                        else:
                            Rmax_ID = int(Rmax_ID[0][0])
                            m = X[Rmax_ID]
                        #print('Rmax_ID = ',Rmax_ID)
                        #print('m = ',[m[0][0],m[0][1]])
                        if m not in self.unavail:
                            Area.setdefault(k, []).append([m[0][0],m[0][1]])
                            self.unavail.append([m[0][0],m[0][1]])
                        else:
                            break
                    else:
                        break
            return Area
        
        #S T E P   1   (C R E A T E   A R E A S)
        
        self.V = {}
        self.A = {}
        self.unavail = []
        k = 0
        #print('Creating Network Areas of '+str(month))
        for i,j in itertools.product(range(self.dimX),range(self.dimY)):
            if ((i*self.dimY)+j) in self.nodes[0,:]:
                ID = np.where(self.nodes[0,:] == ((i*self.dimY)+j))
                ID = int(ID[0])
                #print('ID = ',ID)
                if [i,j] not in self.unavail:
                    while True:
                        nei_1,nei_2,nei_3,nei_4 = gen_cell_neighbours(i, j, i_nan, j_nan)
                        #print('nei_1 = ',nei_1,'nei_2 = ',nei_2,'nei_3 = ',nei_3,'nei_4 = ',nei_4)
                        nei_list = [nei_1, nei_2 ,nei_3, nei_4]
                        nei_corrs = [self.corrs[ID,nei_1[0],nei_1[1]], self.corrs[ID,nei_2[0],nei_2[1]], self.corrs[ID,nei_3[0],nei_3[1]], self.corrs[ID,nei_4[0],nei_4[1]]]
                        nei_max = np.nanmax([self.corrs[ID,nei_1[0],nei_1[1]], self.corrs[ID,nei_2[0],nei_2[1]], self.corrs[ID,nei_3[0],nei_3[1]], self.corrs[ID,nei_4[0],nei_4[1]]])
                        #print('nei_max = ',nei_max)
                        if nei_max > self.tau:
                            nei_max_ID = np.where(nei_corrs==nei_max)
                            if np.shape(nei_max_ID) == 1:
                                nei_max_ID = int(nei_max_ID[0])
                                nei_max_ID = nei_list[nei_max_ID]
                            else:
                                nei_max_ID = int(nei_max_ID[0][0])
                                nei_max_ID = nei_list[nei_max_ID]
                            if ([i,j] not in self.unavail) and ([nei_max_ID[0],nei_max_ID[1]] not in self.unavail):
                                self.A.setdefault(k, []).append([i,j])
                                self.A.setdefault(k, []).append([nei_max_ID[0],nei_max_ID[1]])
                                #print('A = ',self.A)
                                self.unavail.append([i,j])
                                self.unavail.append([nei_max_ID[0],nei_max_ID[1]])
                                #print('unavail (pre expand) = ',self.unavail)
                                self.V = expand(self.A)
                                #print('unavail (post expand) = ',self.unavail)
                                #print('V = ',self.V)
                                k = k + 1
                            else:
                                break
                        else:
                            break
                            
        #print(str(month)+' number of areas, before minimisation = ',len(self.V))
        
        #S T E P   2   (M I N I M I S E   NO.   O F   A R E A S)
        
        self.unavail = []
        while True:
            num_cells = {}
            Anei_Rs = {}
            unavail_neis = []
            #Identify largest area in terms of number of cells
            for k in self.V:
                if self.V[k][0] not in self.unavail:
                    num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
                else:
                    num_cells.setdefault(k, []).append(0)
            max_ID = max(num_cells.items(), key=operator.itemgetter(1))[0]
            if num_cells[max_ID][0] == 0:
                break
            else:
                #print('AreaID = ',max_ID, ', # of cells = ',len(self.V[max_ID]))
                for X in self.V[max_ID]: #for each cell in the currently available largest area
                    nei_1, nei_2, nei_3, nei_4 = gen_cell_neighbours(X[0],X[1], i_nan, j_nan) #generate the cell's available neighbours
                    nei_list = [nei_1, nei_2, nei_3, nei_4]
                    for k in self.V: #search through all other areas in the network   
                        for nei in nei_list: #search through each neighbour of the current cell in largest area
                            R_mean = []
                            if (nei not in self.V[max_ID]) & (nei in self.V[k]) & (nei not in unavail_neis): #if the neighbouring cell belongs to a neighbouring AREA, and is available
                                #print('nei = ',nei,'is in Area ',k,'and is not in Area',max_ID)
                                #print('Area',k,' = ',self.V[k])
                                for i in range(np.shape(self.V[k])[0]):
                                    unavail_neis.append(self.V[k][i])
                                #here make a hypothetical area of the largest area (max_ID) and it's available neighbour (k) to check average correlation    
                                hypoth_area = []
                                for cell in self.V[max_ID]:
                                    hypoth_area.append([cell[0],cell[1]])
                                for cell in self.V[k]:
                                    hypoth_area.append([cell[0],cell[1]])
                                NA_list = []
                                for cell in hypoth_area:
                                    #print(cell)
                                    R = []
                                    ID = np.where(self.nodes[0,:] == (cell[0]*self.dimY)+cell[1])
                                    ID = int(ID[0])
                                    for a in range(np.shape(hypoth_area)[0]):
                                        b = int(hypoth_area[a][0])
                                        c = int(hypoth_area[a][1])
                                        if ([b,c] != [cell[0],cell[1]]) & ([b,c] not in NA_list):
                                            #print('[',b,',',c,']')
                                            R.append(self.corrs[ID,b,c])
                                    NA_list.append([cell[0],cell[1]])
                                    R_mean.append(np.nanmean(R))   
                                if k not in Anei_Rs: 
                                    Anei_Rs.setdefault(k, []).append(np.nanmean(R_mean))
                                #print('Average correlation with Area',max_ID,'and neighbouring Area',k,' = ',Anei_Rs[k])
                try:
                    Anei_Rs_max_ID = max(Anei_Rs.items(), key=operator.itemgetter(1))[0]
                    #print('Maximum correlation with neighbouring area = ',Anei_Rs[Anei_Rs_max_ID][0])
                    if Anei_Rs[Anei_Rs_max_ID][0] > self.tau:
                        #print('ID_pair = ',Anei_Rs_max_ID)
                        temp2 = self.V.pop(Anei_Rs_max_ID, None)
                        for i in temp2:
                            self.V.setdefault(max_ID, []).append([i[0],i[1]])
                    else:
                        for i in range(np.shape(self.V[max_ID])[0]):
                            self.unavail.append(self.V[max_ID][i])
                except ValueError:
                    for i in range(np.shape(self.V[max_ID])[0]):
                        self.unavail.append(self.V[max_ID][i])
                        
        #print(str(month)+' number of areas = ',len(self.V))
        
        num_cells = {}
        for k in self.V:
            num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
        max_ID = max(num_cells.items(), key=operator.itemgetter(1))[0]
        
        num_cells = {}
        for k in self.V:
            if k != max_ID:
                num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
        max_ID2 = max(num_cells.items(), key=operator.itemgetter(1))[0]
        
        #print('Largest Area = #',max_ID,' with ',len(self.V[max_ID]),' cells')
        #print('2nd Largest Area = #',max_ID2,' with ',len(self.V[max_ID2]),' cells')
                
    def intra_links(self, area=None, lat=None):
        """
        compute the anomaly time series associated with
        every node of the network, and subsequently compute
        weighted links (based on covariance) between all of
        these nodes. The strength of each node (also known as
        the weighted degree), is defined as the sum of the
        absolute value of each nodes links. Here the network
        is fully connected, so every node connects to every other
        node
        """
        self.anomaly = {}
        self.links = {}
        self.strength = {}
        self.strengthmap = np.zeros((self.dimX,self.dimY))*np.nan
        if lat is not None:
            scale = np.sqrt(np.cos(np.radians(lat)))
        elif area is not None:
            scale = np.sqrt(area)
        else:
            scale = np.ones((self.dimX,self.dimY))
            
        for A in self.V:
            temp_array = np.zeros(self.data.shape)*np.nan
            for cell in self.V[A]:
                temp_array[cell[0],cell[1],:] = np.multiply(self.data[cell[0],cell[1],:],scale[cell[0],cell[1]])
            self.anomaly[A] = np.nansum(temp_array, axis=(0,1))
            
        for A in self.anomaly:
            sdA = np.std(self.anomaly[A])
            for A2 in self.anomaly:
                sdA2 = np.std(self.anomaly[A2])
                if A2 != A:
                    self.links.setdefault(A, []).append(stats.pearsonr(self.anomaly[A],self.anomaly[A2])[0]*(sdA*sdA2))
                elif A2 == A:
                    self.links.setdefault(A, []).append(0)
            
        for A in self.links:
            absolute_links = []  
            for link in self.links[A]:
                absolute_links.append(abs(link))
            self.strength[A] = np.nansum(absolute_links)
            for cell in self.V[A]:
                self.strengthmap[cell[0],cell[1]] = self.strength[A]
