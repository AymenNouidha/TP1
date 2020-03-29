import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

def feature13(x):
    try:
        if(x.NumberOfFollowers > 0):
            return x.NumerOfFollowings/x.NumberOfFollowers
        else:
            return np.nan
    except:
        return np.nan

def feature14(x):
    try:
        lim = x.TweetID.count()
        if(lim > 1):
            return (((max(x['CreatedAt']) - min(x['CreatedAt'])).total_seconds())/(x.TweetID.count()-1))/(60)
        else:
            return np.nan
    except:
        return np.nan

def feature15(x):
    try:
        lim = x.TweetID.count()
        if(lim > 1):
            df =  (x.loc[1:lim-1, 'CreatedAt']).reset_index(drop=True)
            df2 = (x.loc[0:lim-2, 'CreatedAt']).reset_index(drop=True)
            dfp = ((((df.subtract(df2)).dt.total_seconds()).max(axis = 0)))/60
            return dfp
        else:
            return np.nan
    except:
        return np.nan
def feature16(x):
    try:
        lim = x.TweetID.count()
        if(lim > 1):
            ser = x.loc[0:lim-1, 'Tweet'].tolist()
            count_vectorizer = CountVectorizer(stop_words='english')
            sparse_matrix = count_vectorizer.fit_transform(ser)
            doc_term_matrix = sparse_matrix.todense()
            df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names())
            df1 = cosine_similarity(df)
            return (((df1.sum()).sum())-lim)/(np.square(lim) - lim)
        else:
            return np.nan
    except:
        return np.nan

def fill_nanValues(dataFrame, La_classe):
    scaler = StandardScaler()
    dataFrame.drop(['UserID', 'CreatedAt', 'CollectedAt', 'nombre_total_tweet', 'somme_url_total', 'somme_mention_total'], axis = 1, inplace=True)
    #(dataFrame.loc[:, 'NumerOfFollowings':'similarity']).fillna((dataFrame.loc[:, 'NumerOfFollowings':'similarity']).median(), inplace=True)
    dataFrame.to_csv('interm'+str(La_classe)+'.csv')
    dataFrame.fillna(dataFrame.median(), inplace=True)
    dataFrame.drop_duplicates(inplace = True)
    #dataFrame.loc[:, 'NumerOfFollowings':'similarity'] = (dataFrame.loc[:, 'NumerOfFollowings':'similarity'] - dataFrame.loc[:, 'NumerOfFollowings':'similarity'].mean())/dataFrame.loc[:, 'NumerOfFollowings':'similarity'].std()
    #dataFrame.loc[:,:] = scaler.fit_transform(dataFrame.to_numpy())
    dataFrame['Classe'] = La_classe
    return dataFrame