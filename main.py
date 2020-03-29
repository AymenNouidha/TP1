import numpy as np
import pandas as pd
from conversion import featuresCalc
from customLib  import fill_nanValues

content_polluters_file_path = 'content_polluters.txt'
content_polluters_followings_file_path = 'content_polluters_followings.txt'
content_polluters_tweets_file_path = 'content_polluters_tweets.txt'
legitimate_users_file_path = 'legitimate_users.txt'
legitimate_users_followings_file_path = 'legitimate_users_followings.txt'
legitimate_users_tweets_file_path = 'legitimate_users_tweets.txt'

polluter_df   = featuresCalc(content_polluters_file_path, content_polluters_followings_file_path, content_polluters_tweets_file_path)
polluter_df   = fill_nanValues(polluter_df, 1)
legitimate_df = featuresCalc(legitimate_users_file_path, legitimate_users_followings_file_path, legitimate_users_tweets_file_path)
legitimate_df = fill_nanValues(legitimate_df, 0)
Features_DF   = pd.concat([polluter_df, legitimate_df], axis=0)
Features_DF = Features_DF.reset_index(drop=True)
Features_DF = Features_DF.sample(frac=1, axis=0)
Features_DF.to_csv('Features_DF.csv')
print(Features_DF.head())


 