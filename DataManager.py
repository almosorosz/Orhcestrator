#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pure_functions import W_pred, corr_func, L_pred, smoother, solub, final_value, SED

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize



# In[12]:


class DataManager:
    def __init__(self,LOCS):
        self.LOCS         = LOCS
        self.DESC_best_W   = [2]
        self.DESC_best     = [1, 2]
        self.finaltime     = 10 # minutes
        self.colors       = ['blue','red','green']
        self.conc_corr     = True
        self.fig, self.axT = plt.subplots(2,2,figsize=(10,7))
        self.res           = []

    def plotter(self, E1, T1, WWW_1, LLL_1, SL1, SW1, j, collor='blue'):
        
        self.axT[0,0].plot(T1,E1['H01:External T'],'black',linestyle='--',alpha=0.5)
        ax_FBRM = self.axT[0,0].twinx()
        ax_FBRM.plot(T1,E1['OPC R: Counts '],collor,linestyle='-',alpha=0.7)
        self.axT[0,0].set_xlabel(E1.columns[0][0]+' '+'['+E1.columns[0][1]+']')
        self.axT[0,0].set_ylabel(E1.columns[2][0]+' '+'['+E1.columns[2][1]+']')
        ax_FBRM.set_ylabel(E1.columns[4][0]+' '+'['+E1.columns[4][1]+']')
        self.axT[0,0].grid(True)
        ax_FBRM.set_ylim([0,80000])
        
        self.axT[0,1].plot(T1,E1['H01:External T'],'black',linestyle='--',alpha=0.5)
        
        if self.conc_corr: #correction for UV-shift
            ax_conc = self.axT[0,1].twinx()
            bnds = ((1e-10, 100), (-10, 100))
            x0 = np.array([1.1, 0.002])
            X = [E1['File R: Conc '],solub(E1['H01:External T'])]
            res = minimize(corr_func, x0, args=(X), method='nelder-mead',bounds=bnds,options={'xatol': 1e-20, 'disp': False})
            popt = [res.x[0],res.x[-1]]
            
        
        ax_conc.scatter(T1,popt[0]*E1['File R: Conc ']+popt[1],c=collor,s=4,alpha=0.7)
        ax_conc.plot(T1,solub(E1['H01:External T']))
        self.axT[0,1].set_xlabel(E1.columns[0][0]+' '+'['+E1.columns[0][1]+']')
        self.axT[0,1].set_ylabel(E1.columns[2][0]+' '+'['+E1.columns[2][1]+']')
        ax_conc.set_ylabel(E1.columns[-1][0]+' '+'['+E1.columns[-1][1]+']')
        ax_conc.set_ylim([0.04,0.08])
        self.axT[0,1].grid(True)
    
        mask = (T1 > max(T1) - self.finaltime)
        
        self.axT[1,0].plot(T1,E1['H01:External T'],'black',linestyle='--',alpha=0.2)
        ax_L = self.axT[1,0].twinx()
        ax_L.plot(T1,LLL_1,collor,linestyle='-',alpha=0.1)
        ax_L.plot(T1[mask],LLL_1.iloc[mask,:],collor,linestyle='-',alpha=0.6)    
        [ax_L.plot(SL1[0],SL1[i],collor, linestyle='-',alpha=0.9) for i in [1,2,3]]
        self.axT[1,0].set_xlabel(E1.columns[0][0]+' '+'['+E1.columns[0][1]+']')
        self.axT[1,0].set_ylabel(E1.columns[2][0]+' '+'['+E1.columns[2][1]+']')
        ax_L.set_ylim([0,300])
        ax_L.set_ylabel('Length'+' '+'['+E1.columns[5][1]+']')
        self.axT[1,0].grid(True)
        
        ax_L.text(plt.xlim()[1]*(0.3+j*0.25),plt.ylim()[1]*1.01,f'Exp. no:{j : .0f}', color=collor)
        
        self.axT[1,1].plot(T1,E1['H01:External T'],'black',linestyle='--',alpha=0.5)
        ax_W = self.axT[1,1].twinx()
        ax_W.plot(T1,WWW_1,collor,linestyle='-',alpha=0.3)
        ax_W.plot(T1[mask],WWW_1.iloc[mask,:],collor,linestyle='-',alpha=0.9)    
        [ax_W.plot(SW1[0],SW1[i],collor, linestyle='-',alpha=0.9) for i in [1,2,3]]
        self.axT[1,1].set_xlabel(E1.columns[0][0]+' '+'['+E1.columns[0][1]+']')
        self.axT[1,1].set_ylabel(E1.columns[2][0]+' '+'['+E1.columns[2][1]+']')
        ax_W.set_ylabel('Width'+' '+'['+E1.columns[9][1]+']')
        self.axT[1,1].grid(True)
        ax_W.set_ylim([0,30])
        self.fig.tight_layout()
        
        pass

    def data_preprocess(self,E1,k):

        T1 = (E1['Time']).values.reshape(1,-1)[0] - (E1['Time'][0:1]).values[0]
        WWW_1 = pd.DataFrame(W_pred(np.array([E1[['OPC R: W10 ','OPC R: W50 ','OPC R: W90 ']].iloc[:,self.DESC_best_W]]))[0][:,:])
        LLL_1 = pd.DataFrame(L_pred(np.array([E1[['OPC R: L10 ','OPC R: L50 ','OPC R: L90 ']].iloc[:,self.DESC_best]]))[0][:,:])
        SL1 = [smoother(LLL_1,T1,2,k)[0],smoother(LLL_1,T1,0,k)[1],smoother(LLL_1,T1,1,k)[1],smoother(LLL_1,T1,2,k)[1]]
        SW1 = [smoother(WWW_1,T1,2,k)[0],smoother(WWW_1,T1,0,k)[1],smoother(WWW_1,T1,1,k)[1],smoother(WWW_1,T1,2,k)[1]]
        
        return T1, WWW_1, LLL_1, SL1, SW1
    
    def evaluator(self):
        for j,i in enumerate(self.LOCS):
            E1 = pd.read_csv(i, header = [0, 1])
            T1, WWW_1, LLL_1, SL1, SW1 = self.data_preprocess(E1,70)
            print(self.colors[j%3])
            self.plotter(E1,T1, WWW_1, LLL_1, SL1, SW1, j, collor=self.colors[j%3])
            #print(SED(final_value(T1, LLL_1, self.finaltime),final_value(T1, WWW_1, self.finaltime)))
            VAL = pd.DataFrame({'Length':final_value(T1, WWW_1, self.finaltime),'Width': final_value(T1, LLL_1, self.finaltime), 'SED': SED(final_value(T1, LLL_1, self.finaltime),final_value(T1, WWW_1, self.finaltime))})        
            self.res.append(VAL)
        return self.res
        


# In[13]:


loc1 = r"D:\1. Doktori\1. Aktív projktek\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\6. Purdue\1. Sandia - Resveratrol Data\New_omnibus\Re_ repr_Omni_1126\expttsd2.csv"
loc2 = r"D:\1. Doktori\1. Aktív projktek\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\6. Purdue\1. Sandia - Resveratrol Data\New_omnibus\Re_ repr_Omni_1126\expttsd4.csv"
loc3 = r"D:\1. Doktori\1. Aktív projktek\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\6. Purdue\1. Sandia - Resveratrol Data\New_omnibus\Re_ repr_Omni_1126\expttsd6.csv"
loc4 = r"D:\1. Doktori\1. Aktív projktek\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\6. Purdue\1. Sandia - Resveratrol Data\New_omnibus\repr_Omni_1126\expttsd2.csv"
loc5 = r"D:\1. Doktori\1. Aktív projktek\OneDrive - Budapesti Műszaki és Gazdaságtudományi Egyetem\6. Purdue\1. Sandia - Resveratrol Data\New_omnibus\repr_Omni_1126\expttsd4.csv"


# In[14]:


LOCS1 = [loc1, loc2, loc3]


R = DataManager(LOCS1)
R.evaluator()


# In[15]:


LOCS2 = [loc4, loc5]
F = DataManager(LOCS2)
F.evaluator()


# In[16]:


# next steps: create a dataframe with all read data and convert it to .mat file and append additional experimental data


# In[43]:


from scipy.io import loadmat
from scipy.io import savemat


data = loadmat(r"C:\Users\user\12. OMNIBUS-IMED integration\2025_12_10_init_models.mat", squeeze_me=True, struct_as_record=True)
#savemat(r"C:\Users\user\modified.mat", data, do_compression=False)


# In[97]:


fig, ax = plt.subplots(1,data['No'],figsize=(12,2) )
for i in range(data['No']):
    
    ax[i].plot(data['ExpData'][i][0][0],data['ExpData'][i][0][1],label='c',color='black')
    ax[i].plot(data['ExpData'][i][4][0],solub(data['ExpData'][i][4][1]),'-.',label='sol',color='grey')
    ax[i].set_xlabel("Time [s]")
    ax[i].set_ylim([0.04,0.1])
    ax[i].set_title(f'Exp. no {i+1:.0f}')
    axT = ax[i].twinx()
    axT.plot(data['ExpData'][i][4][0],data['ExpData'][i][4][1],'--',color='red',label='T')
    axT.yaxis.label.set_color('red')
    axT.tick_params(axis='y', colors='red')
    axT.set_ylim([0,60])
    
fig.tight_layout()


# In[83]:


def in_c(M,S,s):
    return M/(M+M*(s/100)+S)

