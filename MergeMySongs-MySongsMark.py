import pandas as pd

df_songs = pd.read_csv('SongsFeatures.csv')
df_songs = df_songs.drop(['name','Unnamed: 0'],axis=1)

df_songsMark = pd.read_csv('SongsMark.csv')
df_songsMark = df_songsMark.drop(['Unnamed: 0'],axis=1)

df_merged = df_songs.merge(df_songsMark, how='inner', on='id')
df_merged = df_merged.drop(['id'],axis=1)

df_merged.to_csv('SongsDataset.csv')