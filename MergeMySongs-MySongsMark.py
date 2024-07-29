import pandas as pd

df_songs = pd.read_csv('Songs.csv')

df_songsMark = pd.read_csv('SongsMark.csv')

df_merged = df_songs.merge(df_songsMark, how='inner', on='id')