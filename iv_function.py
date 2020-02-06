
# import packages
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string
import os

max_bin = 20

# define a binning function
def mono_bin(Y, X, n = max_bin):
    force_bin = 3
    
    df = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df[['X','Y']][df.X.isnull()]
    notmiss = df[['X','Y']][df.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df[['X','Y']][df.X.isnull()]
    notmiss = df[['X','Y']][df.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def information_value(df, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df[i], np.number) and len(Series.unique(df[i])) > 2:
                conv = mono_bin(target, df[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    
    var_bin_freq = iv_df.groupby(['VAR_NAME']).size().reset_index(name='counts')
    var_bin_freq.columns = ['VAR_NAME', 'BINS']
    
    ### Merge
    iv = pd.merge(iv,
                  var_bin_freq,
                  on = 'VAR_NAME',how = 'left')
    iv = iv.sort_values('IV', ascending=False)
    iv = iv.reset_index(drop = True)
    return(iv)

#####Execute the function
#iv = information_value(df = df,target = df.target)


## 

def fn_biz_viz(X,y,Target,output=os.getcwd()):
    X['decile']=pd.qcut(X[y], 10, labels=False)
    Rank=X.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[Target]),
        np.size(x[Target][x[Target]==0]),
        ],
        index=(["min_resp","max_resp","avg_resp","cnt","cnt_resp","cnt_non_resp"])
        )).reset_index()
    Rank["rrate"]=round(Rank["cnt_resp"]*100/Rank["cnt"],3)
    Rank["cum_resp"]=np.cumsum(Rank["cnt_resp"])
    Rank["cum_non_resp"]=np.cumsum(Rank["cnt_non_resp"])
    Rank["cum_rel_resp"]=round(Rank["cum_resp"]*100/np.sum(Rank["cnt_resp"]),3)
    Rank["cum_rel_non_resp"]=round(Rank["cum_non_resp"]*100/np.sum(Rank["cnt_non_resp"]),3)
    Rank["KS"] = abs(Rank["cum_rel_resp"] - Rank["cum_rel_non_resp"])
    Rank["rrate"] =Rank["rrate"].astype(str) + '%'
    Rank["cum_rel_resp"]= Rank["cum_rel_resp"].astype(str) + '%'
    Rank["cum_rel_non_resp"]= Rank["cum_rel_non_resp"].astype(str) + '%'
    Rank["KS"] = Rank["KS"].astype(str) + '%'
    Rank
    file_name = output+"/"+y+".csv"
    Rank.to_csv(file_name, index=None)
    return(Rank)

    
#fn_biz_viz(tmp,"No_OF_CR_TXNS","Target",r"E:\K2_Analytics\Training_ppts_Python\logistic\New folder")
    



def RRate(X,y,Target,output=os.getcwd()):
    X['decile']=pd.qcut(X[y], 10, labels=False)
    Rank=X.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[Target]),
        np.size(x[Target][x[Target]==0]),
        ],
        index=(["min_resp","max_resp","avg_resp","cnt","cnt_resp","cnt_non_resp"])
        )).reset_index()
    Rank=Rank.sort_values(by='decile',ascending=False)
    Rank["rrate"]=round(Rank["cnt_resp"]*100/Rank["cnt"],3)
    Rank["cum_resp"]=np.cumsum(Rank["cnt_resp"])
    Rank["cum_non_resp"]=np.cumsum(Rank["cnt_non_resp"])
    Rank["cum_rel_resp"]=round(Rank["cum_resp"]*100/np.sum(Rank["cnt_resp"]),3)
    Rank["cum_rel_non_resp"]=round(Rank["cum_non_resp"]*100/np.sum(Rank["cnt_non_resp"]),3)
    Rank["KS"] = abs(Rank["cum_rel_resp"] - Rank["cum_rel_non_resp"])
    Rank["rrate"] =Rank["rrate"].astype(str) + '%'
    Rank["cum_rel_resp"]= Rank["cum_rel_resp"].astype(str) + '%'
    Rank["cum_rel_non_resp"]= Rank["cum_rel_non_resp"].astype(str) + '%'
    Rank["KS"] = Rank["KS"].astype(str) + '%'
    Rank
    return(Rank)
