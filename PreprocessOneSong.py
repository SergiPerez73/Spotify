import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import PreprocessDataset
import numpy as np
from transformers import BertTokenizer, BertModel
import umap


def createSpotifyApiSession(token, client_id,client_secret):
    sp = spotipy.Spotify(auth=token,
                        auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                            client_secret=client_secret))
    
    return sp

def getWantedFeatures(track_dict):
    track_features = {key:[track_dict[key]] for key in ['id','duration_ms','explicit','name','popularity']}
    track_features['album_type'] = [track_dict['album']['album_type']]
    track_features['album_name'] = [track_dict['album']['name']]
    track_features['release_date'] = [track_dict['album']['release_date']]
    track_features['artist_name'] = [track_dict['artists'][0]['name']]
    
    return track_features


def getSongsFeatures(sp,track_ids):
    track_features = {}
    offset = 0
    for track_id in track_ids:
        track_dict = sp.track(track_id)
        if offset == 0:
            track_features = getWantedFeatures(track_dict)
        else:
            track_features_aux = getWantedFeatures(track_dict)
            for key in track_features:
                track_features[key] = track_features[key] + track_features_aux[key]

        offset +=1
        print(offset)

    df_SongsFeatures = pd.DataFrame(track_features)
    df_SongsFeatures.to_csv('OneSong.csv')
    return df_SongsFeatures

def addEmbeddingsTodf(df,df_OneSong,col):
    words = df.loc[:,col]
    outputs = (PreprocessDataset.getEmbedding(words[0]))
    for i in range(1,len(words)):
        outputs = np.append(outputs,(PreprocessDataset.getEmbedding(words[i])),axis=0)
        #print(i)
    
    words = df_OneSong.loc[:,col]
    outputs_OneSong = (PreprocessDataset.getEmbedding(words[0]))
    for i in range(1,len(words)):
        outputs_OneSong = np.append(outputs_OneSong,(PreprocessDataset.getEmbedding(words[i])),axis=0)

    umap_model = umap.UMAP(n_components=3, random_state=42)
    embeddings_2d = umap_model.fit_transform(outputs)

    embeddings_2d_OneSong = umap_model.transform(outputs_OneSong)


    emb_df_OneSong = pd.DataFrame(embeddings_2d_OneSong, columns=[col+'1', col+'2', col+'3'])
    df_OneSong = pd.concat([df_OneSong, emb_df_OneSong], axis=1)
    
    print('Finished ',col)

    return df_OneSong

if __name__ == "__main__":
    sp = createSpotifyApiSession("BQAkWRmETTwKhvP4IM-YjUaySowh-J3b0WnM6-eHyW_GwI9lnNc8IWmqc2nge3c5dc43mVRGvmWz9k5HeqKN0EO8yNIisVRl3t42-SQEEmy7AU3Kfnf8nh760Zw4K55LRurG-wNb-hag0Nw2Uyr0YmzqhuwfLtZ2thYJOmO_WTypbHiKbjn2x0lswgYLkruEEgJ6_jMEV7J9ZRmkfyA"
                                 ,'16c7c999caa041fe96aa9e01c2abf86d','4e10f9170b4444f2889bab3497806147')

    track_ids = ['7u0Mn9qAgZxcSWm0db2PaG','3hfmh1XIlJp2Uis4kWboqJ','0DIcssPpatAMqFXLZCxZMN','4n93SK7dQsQVu9BM5QzvAx','0nky5lP13IyrlQEGEzGQZt']
    df_OneSong = getSongsFeatures(sp,track_ids)

    df = pd.read_csv('SongsDataset.csv')

    #Preprocess explicit
    type_map = {False: 0.0, True: 1.0}
    df_OneSong['explicit'] = df_OneSong["explicit"].map(type_map)

    #Preprocess album_type
    type_map = {'single': 0.0, 'album': 1.0, 'compilation': 1.0}
    df_OneSong['album_type'] = df_OneSong["album_type"].map(type_map)

    #Preprocess album_name
    df_OneSong = addEmbeddingsTodf(df,df_OneSong,'album_name')

    #Preprocess release_date
    df_OneSong = addEmbeddingsTodf(df,df_OneSong,'release_date')

    #Preprocess artist_name
    df_OneSong = addEmbeddingsTodf(df,df_OneSong,'artist_name')

    #Preprocess name
    df_OneSong = addEmbeddingsTodf(df,df_OneSong,'name')

    df_OneSong = df_OneSong.drop(['album_name','release_date','artist_name','name'],axis=1)

    df_OneSong.to_csv('OneSongPreprocessed.csv')