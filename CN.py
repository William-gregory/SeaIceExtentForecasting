## This code has been generated to create a network framework based on geospatial time series data. The basis of the theory
## can be found in Fountalis, 2014. This code will cluster time series data into geographic areas which are considered to be
## homogeneous in terms of the provided climate field (in this case sea ice concentration). After which, statistical
## links between these areas can be derived, namely covariance.
##
## Author: William Gregory
## Last updated: 19/03/2019

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
    def __init__(self, dimX=0, dimY=0, nodes=[], corrs=[], tau=0, V={}, A={}, unavail=[], anomaly={}, links={}, strength={}):
        self.R = np.zeros((dimX,dimY))
        self.nodes = nodes
        self.corrs = corrs
        self.tau = tau
        self.V = V
        self.A = A
        self.unavail = unavail
        self.anomaly = anomaly
        self.links = links
        self.strength = strength
    
    def cell_level(self, data, month, tag, datapath):
        dimX = self.R.shape[0]
        dimY = self.R.shape[1]
        print('Creating cell-level network of SIC '+str(month))
        print(datetime.datetime.now())
        if not os.path.exists(datapath+'/correlations_'+str(month)+str(tag)):
            os.makedirs(datapath+'/correlations_'+str(month)+str(tag))
        else:
            print('Correlations already exist! Terminating job so as not to overwrite files')
            return
        count = -1
        count_array = []
        files = range(dimX*dimY)
        nodes = dimX*dimY
        for i,j in itertools.product(range(dimX),range(dimY)):
            if max(abs(data[i,j,range(np.shape(data)[2])])) > 0:
                count = count + 1
                count_array.append(count)
                node = i*dimY + j
                for node1 in range(nodes):
                    if node1 % dimY == 0: #These are all the grid cells in column 0
                        node1_row = int(node1/dimY)
                        node1_col = 0
                    else:
                        node1_row = int(math.floor(node1/dimY))
                        node1_col = int(node1 % dimY)  
                    if (node1 != node) & (max(abs(data[node1_row,node1_col,range(np.shape(data)[2])])) > 0):
                        l = -1
                        A = [0]*(np.shape(data)[2])
                        B = [0]*(np.shape(data)[2])
                        for k in range(np.shape(data)[2]):
                            if (~np.isnan(data[i,j,k])) & (~np.isnan(data[node1_row,node1_col,k])):
                                l = l + 1
                                A[l] = data[i,j,k]
                                B[l] = data[node1_row,node1_col,k]
                        if l >= 0:
                            R = stats.pearsonr(A,B)[0]
                            self.R[node1_row,node1_col] = R
                        else:
                            self.R[node1_row,node1_col] = np.nan
                    else:
                        self.R[node1_row,node1_col] = np.nan
                Corr = np.reshape(self.R[:,:],(dimX,dimY))
                np.savetxt(datapath+'/correlations_'+str(month)+str(tag)+'/'+str(files[count]), Corr, delimiter='\t')          
            else:
                count = count + 1
        np.savetxt(datapath+'/correlations_'+str(month)+str(tag)+'/nodes.txt',count_array,fmt='%i',newline='\n')
        print(datetime.datetime.now())
        print('Done!')
    
    def tau(self, data, month, alpha, tag, datapath):
        self.corrs = []
        self.nodes = []
        self.V = {}
        self.A = {}
        #print('Reading correlation files')
        self.nodes.append(np.genfromtxt(datapath+'/correlations_'+str(month)+str(tag)+'/nodes.txt'))
        self.nodes = np.array(self.nodes) 
        nodes_len = np.arange(self.nodes[0].shape[0])
        for i in nodes_len:
            self.corrs.append(np.genfromtxt(datapath+'/correlations_'+str(month)+str(tag)+'/'+str("%02d"%self.nodes[0,i]), autostrip=True)) 
        self.corrs = np.array(self.corrs)
        
        #print('Running T-tests')
        df = np.shape(data)[2] - 2
        #print('Degrees of Freedom: ',df)
        copy_corrs = np.copy(self.corrs)
        copy_corrs[copy_corrs<0] = np.nan
        flat_corrs = copy_corrs.ravel()
        corrs_alpha = []
        for R in flat_corrs:
            if abs(R) == 1:
                P = 0
            elif ~np.isnan(R):
                T = (R)*np.sqrt(df/(1 - (R**2)))
                P = stats.t.sf(T, df)
                if P < alpha:
                    corrs_alpha.append(R)
        
        #print('Minimum Significant Correlation = ',np.nanmin(corrs_alpha))
        self.tau = np.nanmean(corrs_alpha)

        print('SIC '+str(month)+' tau = ',self.tau)
    
    def area_level(self, month):
        dimX = self.R.shape[0]
        dimY = self.R.shape[1]
        
        def gen_cell_neighbours(i, j):
            if [i-1,  j] not in self.unavail:
                nei_1 = [i-1,  j] if 0 <=   j <= dimY-1 and 0 <= i-1 <= dimX-1 else [0,0]
            else:
                nei_1 = [0,0]
            if [i+1,  j] not in self.unavail:
                nei_2 = [i+1,  j] if 0 <=   j <= dimY-1 and 0 <= i+1 <= dimX-1 else [0,0]
            else:
                nei_2 = [0,0]
            if [  i,j-1] not in self.unavail:
                nei_3 = [  i,j-1] if 0 <= j-1 <= dimY-1 and 0 <=   i <= dimX-1 else [0,0]
            else:
                nei_3 = [0,0]
            if [  i,j+1] not in self.unavail:
                nei_4 = [  i,j+1] if 0 <= j+1 <= dimY-1 and 0 <=   i <= dimX-1 else [0,0]
            else:
                nei_4 = [0,0]
            return nei_1, nei_2, nei_3, nei_4
        
        def gen_area_neighbours(Area):
            Anei_1 = []
            Anei_2 = []
            Anei_3 = []
            Anei_4 = []
            for A in range(np.shape(Area[k])[0]):
                if [Area[k][A][0]-1,Area[k][A][1]] not in self.unavail:
                    Anei_1.append([Area[k][A][0]-1,Area[k][A][1]] if 0 <=   Area[k][A][1] <= dimY-1 and 0 <= Area[k][A][0]-1 <= dimX-1 else [0,0])
                if [Area[k][A][0]+1,Area[k][A][1]] not in self.unavail:
                    Anei_2.append([Area[k][A][0]+1,Area[k][A][1]] if 0 <=   Area[k][A][1] <= dimY-1 and 0 <= Area[k][A][0]+1 <= dimX-1 else [0,0])
                if [Area[k][A][0],Area[k][A][1]-1] not in self.unavail:
                    Anei_3.append([Area[k][A][0],Area[k][A][1]-1] if 0 <=   Area[k][A][1]-1 <= dimY-1 and 0 <= Area[k][A][0] <= dimX-1 else [0,0])
                if [Area[k][A][0],Area[k][A][1]+1] not in self.unavail:
                    Anei_4.append([Area[k][A][0],Area[k][A][1]+1] if 0 <=   Area[k][A][1]+1 <= dimY-1 and 0 <= Area[k][A][0] <= dimX-1 else [0,0])
            return Anei_1, Anei_2, Anei_3, Anei_4

        def area_max_correlation(area_neighbours, Area):
            R_mean = []
            X = []
            for nei in area_neighbours:
                R = []
                if ((nei[0][0]*dimY)+nei[0][1]) in self.nodes[0,:]:
                    #print('Anei = ',nei)
                    X.append(nei)
                    ID_new = np.where(self.nodes[0,:] == ((nei[0][0]*dimY)+nei[0][1]))
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
                Anei_1, Anei_2, Anei_3, Anei_4 = gen_area_neighbours(Area)
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
        #print('Creating Network Areas of SIC '+str(month))
        for i,j in itertools.product(range(dimX),range(dimY)):
            if ((i*dimY)+j) in self.nodes[0,:]:
                ID = np.where(self.nodes[0,:] == ((i*dimY)+j))
                ID = int(ID[0])
                #print('ID = ',ID)
                if [i,j] not in self.unavail:
                    while True:
                        nei_1,nei_2,nei_3,nei_4 = gen_cell_neighbours(i, j)
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
                            
        #print('SIC '+str(month)+' number of areas, before minimisation = ',len(self.V))
        
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
                    nei_1, nei_2, nei_3, nei_4 = gen_cell_neighbours(X[0],X[1]) #generate the cell's available neighbours
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
                                    ID = np.where(self.nodes[0,:] == (cell[0]*dimY)+cell[1])
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
                        
        print('SIC '+str(month)+' number of areas = ',len(self.V))
        
        num_cells = {}
        for k in self.V:
            num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
        max_ID = max(num_cells.items(), key=operator.itemgetter(1))[0]
        
        num_cells = {}
        for k in self.V:
            if k != max_ID:
                num_cells.setdefault(k, []).append(np.shape(self.V[k])[0])
        max_ID2 = max(num_cells.items(), key=operator.itemgetter(1))[0]
        
        print('Largest Area = #',max_ID,' with ',len(self.V[max_ID]),' cells')
        #print('2nd Largest Area = #',max_ID2,' with ',len(self.V[max_ID2]),' cells')
                
    def intra_links(self, data, month, lat):
        dimX = self.R.shape[0]
        dimY = self.R.shape[1]
        lats_rads = (lat*math.pi)/180
        cos_lats = np.zeros((dimX,dimY))
        for i in range(dimX):
            for j in range(dimY):
                cos_lats[i,j] = math.cos(lats_rads[i,j])
        self.anomaly = {}
        self.links = {}
        self.strength = {}
        Ar = {}
        Ar = {}
        for A in self.V:
            temp_array = np.zeros((dimX,dimY,np.shape(data)[2]))
            temp_array[temp_array==0] = np.nan
            for cell in self.V[A]:
                for i,j in itertools.product(range(dimX),range(dimY)):
                    if [i,j] == [cell[0],cell[1]]:
                        for y in range(np.shape(data)[2]):
                            temp_array[cell[0],cell[1],y] = np.multiply(data[cell[0],cell[1],y],cos_lats[cell[0],cell[1]])
            Ar.setdefault(A, []).append(temp_array)
            temp = np.nansum(Ar[A][0],0)
            temp = np.nansum(temp,0)
            #temp = np.nanmean(Ar[A][0],0)
            #temp = np.nanmean(temp,0)
            self.anomaly.setdefault(A, []).append(temp)
          
        for A in self.anomaly: 
            SD1 = np.std(self.anomaly[A][0])
            for A2 in self.anomaly:
                if A2 != A:
                    SD2 = np.std(self.anomaly[A2][0])
                    SDs = np.multiply(SD1,SD2)
                    R = stats.pearsonr(self.anomaly[A][0],self.anomaly[A2][0])[0]
                    self.links.setdefault(A, []).append(np.multiply(SDs,R)) #covariance
                elif A2 == A:
                    self.links.setdefault(A, []).append(100)
              
        for A in self.links:
            absolute = []  
            for i in self.links[A]:
                if i != 100:
                    absolute.append(abs(i))
            self.strength.setdefault(A, []).append(np.nansum(absolute))
        max_ID = max(self.strength.items(), key=operator.itemgetter(1))[0]
        #print('SIC '+str(month)+' area with highest strength = #',max_ID,'with strength of ',self.strength[max_ID])