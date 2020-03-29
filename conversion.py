import numpy as np
import pandas as pd 
from customLib import feature13, feature14, feature15, feature16

def featuresCalc(cpPath, cpfPath, cptPath):
    # Upload the Users DataBase to a DataFrame
    cp_df = pd.read_table(cpPath, header=None, names=["UserID", "CreatedAt", "CollectedAt", "NumerOfFollowings", "NumberOfFollowers",
                                                      "NumberOfTweets", "LengthOfScreenName", "LengthOfDescriptionInUserProfile"])
    
    cp_df.dropna(inplace=True)
    # Transformer les DateTime String en DateTime Object
    cp_df.CreatedAt = pd.to_datetime(cp_df.CreatedAt)
    cp_df.CollectedAt = pd.to_datetime(cp_df.CollectedAt)
    # Ordonner la DataFrame selont le UserID
    cp_df.sort_values('UserID', inplace=True)
    cp_df = cp_df.reset_index(drop=True)
    # Upload the Followings DataBase to a DataFrame
    cpf_df = pd.read_table(cpfPath, header=None, names=['UserID', 'SeriesOfNumberOfFollowings'])
    cpf_df.dropna(inplace=True)
    cpf_df.sort_values('UserID', inplace=True)
    # W=Nous avons besoin de cette DataFrame pour calculer la STD et extraire le MAX et le MIN
    cpf_df_prime = pd.read_csv(cpfPath, sep='[\t,]', engine='python', header=None)
    cpf_df_prime.dropna(inplace=True)
    # Upload the Tweets DataBase to a DataFrame
    cpt_df = pd.read_table(cptPath, header=None, names=["UserID", "TweetID", "Tweet", "CreatedAt"])
    cpt_df.dropna(inplace=True)
    # Transformer les DateTime String en DateTime Object
    cpt_df.CreatedAt = pd.to_datetime(cpt_df.CreatedAt)
    cpt_df.sort_values(['UserID', 'CreatedAt'], inplace=True)
    cpt_df = cpt_df.reset_index()

    # On a que 5 Tweet manquants donc on les supprime
    cpt_df.dropna(subset=['Tweet'], axis=0, inplace=True)
    cp_df = cp_df.loc[cp_df.UserID.isin(cpt_df.UserID),:]
    cpf_df = cpf_df.loc[cpf_df.UserID.isin(cpt_df.UserID),:]
    # Ajouter une colonne à la DataFrame cp_df contenant le rapport NumerOfFollowings par NumberOfFollowers
    cp_df['rapport_following_followers'] = cp_df.apply(lambda x: feature13(x), axis=1) #NumerOfFollowings/cp_df.NumberOfFollowers
    # Ajouter une colonne à la DataFrame cp_df contenant le nombre de jours depuis la création du compte Tweeter
    cp_df['age_account_jours'] = ((cp_df.CollectedAt - cp_df.CreatedAt).dt.total_seconds())/(60*60*24)
    
    # Ajouter une colonne contenant la Date sans la partie hh:mm:ss pour pour pouvoir l'utiliser pour compter les tweet par jour
    cpt_df['short_date'] = (cpt_df.loc[:,'CreatedAt']).dt.date
    # Creer une Series temporaire contenant la moyenne des tweets envoyées par journée active par chaque utilisateur
    dfp = (cpt_df.groupby('UserID')['short_date'].value_counts()).unstack().mean(axis=1)  
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('moyenne_tweet_jour_actif').reset_index()#drop=TTrue
    # Ajouter le Feature moyenne_tweet_jour_actif à la DDataFrame cp_df
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    #cp_df = pd.concat([cp_df, dfp.reset_index()], axis=1, join='inner').drop(['index'], axis=1)
    # Creer une Series temporaire contenant la somme des tweets envoyées chaque journée par chaque utilisateur
    dfp = (cpt_df.groupby('UserID')['TweetID'].count())#.unstack().fillna(0)#.sum(axis=1)
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('nombre_total_tweet').reset_index()#drop=True
    '''cp_df.to_csv('cp_df.csv')
    cpt_df.to_csv('cpt_df.csv')
    cpf_df.to_csv('cpf_df.csv')
    dfp.to_csv('somme_tweet.csv')
    print('finish')'''
    print(dfp.head())
    # Ajouter le Feature nombre_total_tweet à la DataFrame cp_df
    
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    
    cp_df['moyenne_tweet_age_account'] = cp_df['nombre_total_tweet'] / cp_df['age_account_jours']
    #cp_df = pd.concat([cp_df, dfp.reset_index()], axis=1, join='inner').drop(['index'], axis=1)
    # Creer une Series temporaire contenant la somme des URL dans chaque Tweet
    #dfp = cpt_df.groupby('TweetID').Tweet.apply(lambda x: x.str.count(r'[https|ftp]?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'))
    dfp = (cpt_df.groupby(['UserID', 'TweetID'])).Tweet.apply(lambda x: x.str.count(r'[https|ftp]?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'))
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('nombre_url_tweet').reset_index(drop=True)
    # Ajouter le Feature nombre_url_tweet à la DataFrame cpt_df
    cpt_df = pd.concat([cpt_df, dfp], join='inner', axis=1)
    # Creer une Series temporaire contenant le nombre moyen d'url de chaque User par nombre de Tweet totale
    dfp = cpt_df.groupby('UserID').nombre_url_tweet.mean()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('moyenne_url_par_nombre_tweet_total').reset_index()
    # Ajouter le Feature moyenne_url_par_nombre_tweet_total à la DataFrame cp_df
    
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    
    #Creer une Series temporaire contenant le nombre moyen url de chaque User par nombre de Tweet contenant des urls
    dfp = cpt_df.loc[cpt_df.nombre_url_tweet != 0].groupby('UserID').nombre_url_tweet.mean()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('moyenne_url_par_tweet_contenant_urls').reset_index()
    # Ajouter le Feature moyenne_url_par_tweet_contenant_urls à la DataFrame cp_df
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    #cp_df.loc[~cp_df.UserID.isin(dfp.UserID), 'moyenne_url_par_tweet_contenant_urls'] = np.nan
    # Creer une Series temporaire contenant la somme d'urls de chaque User
    dfp = cpt_df.groupby('UserID').nombre_url_tweet.sum()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('somme_url_total').reset_index()
    # Ajouter le Feature somme_url_total à la DataFrame cp_df
    
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    
    # Ajouter le Feature moyenne_url_tweet_par_age_account
    cp_df['moyenne_url_tweet_par_age_account'] = cp_df.somme_url_total / cp_df.age_account_jours
    # Creer une Series temporaire contenant la somme des @ dans chaque Tweet
    dfp = (cpt_df.groupby(['UserID', 'TweetID'])).Tweet.apply(lambda x: x.str.count(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)'))
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('nombre_mention_tweet').reset_index(drop=True)
    # Ajouter le Feature nombre_mention_tweet à la DataFrame cpt_df
    cpt_df = pd.concat([cpt_df, dfp], join='inner', axis=1)
    # Creer une Series temporaire contenant la moyenne des @ dans chaque Tweet de chaque User par nombre de Tweet totale
    dfp = cpt_df.groupby('UserID').nombre_mention_tweet.mean()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('moyenne_mention_par_nombre_tweet_total').reset_index()
    # Ajouter le Feature moyenne_mention_par_nombre_tweet_total à la DataFrame cp_df
    
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
   
    # Creer une Series temporaire contenant le nombre moyen d'@ de chaque User par nombre de Tweet contenant des @
    '''dfp = cpt_df.loc[cpt_df.nombre_mention_tweet != 0].groupby('UserID').nombre_mention_tweet.mean()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('moyenne_mention_par_tweet_contenant_mentions').reset_index()
    # Ajouter le Feature moyenne_mention_par_tweet_contenant_mentions à la DataFrame cp_df
    print('User uniqueB6' + str(len(cp_df.UserID.unique())))
    cp_df = pd.merge(cp_df, dfp, on='UserID',validate='1:1')
    print('User uniqueA6' + str(len(cp_df.UserID.unique())))'''
    # Creer une Series temporaire contenant la somme d'@ de chaque User
    dfp = cpt_df.groupby('UserID').nombre_mention_tweet.sum()
    # Transformer la Series dfp en une DataFrame
    dfp = dfp.to_frame('somme_mention_total').reset_index()
    # Ajouter le Feature somme_mention_total à la DataFrame cp_df
    
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    
    # Ajouter le Feature moyenne_url_tweet_par_age_account
    cp_df['moyenne_mention_tweet_par_age_account'] = cp_df.somme_mention_total / cp_df.age_account_jours
    # Ajouter le Feature maxFollowing à la DataFrame cpf_df
    cpf_df['maxFollowing'] = cpf_df_prime.loc[:,1:].max(axis=1)
    # Ajouter le Feature minFollowing à la DataFrame cpf_df
    cpf_df['minFollowing'] = cpf_df_prime.loc[:,1:].min(axis=1)
    # Ajouter le Feature stdFollowing à la DataFrame cpf_df
    cpf_df['stdFollowing'] = cpf_df_prime.loc[:,1:].std(axis=1)
    cp_df = pd.merge(cp_df, cpf_df.loc[:, ['UserID', 'maxFollowing', 'minFollowing', 'stdFollowing']], how='left', on='UserID')
    
    # Creer une Series temporaire contenant la moyenne de temps entre deux tweet consécutives de chaque User
    dfp   = cpt_df.groupby('UserID').apply(lambda x: feature14(x))
    # Transformer la Series dfp en une DataFrame
    dfp   = dfp.to_frame('temps_moyen__minute_entre_tweet_consecutifs').reset_index()
    # Ajouter le Feature temps_moyen__minute_entre_tweet_consecutifs à la DataFrame cp_df
   
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    
    
    #dfp = cpt_df.sort_values(['UserID', 'CreatedAt']).groupby('UserID').apply(lambda x: feature15(x))
    dfp = cpt_df.groupby('UserID').apply(lambda x: feature15(x))
    dfp = dfp.to_frame('Periode_max_entre_deux_Tweet').reset_index()
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')

    #dfp = cpt_df.sort_values('UserID').groupby('UserID').apply(lambda x: feature16(x))
    dfp = cpt_df.groupby('UserID').apply(lambda x: feature16(x))
    dfp = dfp.to_frame('similarity').reset_index()
  
    cp_df = pd.merge(cp_df, dfp, how='left', on='UserID')
    return cp_df