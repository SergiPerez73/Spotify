import pandas as pd
import PreprocessDataset
import numpy as np
import umap
import CreateDataset

def getSongsFeatures(sp,track_ids):
    track_features = {}
    offset = 0
    for track_id in track_ids:
        track_dict = sp.track(track_id)
        if offset == 0:
            track_features = CreateDataset.getWantedFeatures(track_dict)
        else:
            track_features_aux = CreateDataset.getWantedFeatures(track_dict)
            for key in track_features:
                track_features[key] = track_features[key] + track_features_aux[key]

        offset +=1
        print(offset)

    df_SongsFeatures = pd.DataFrame(track_features)
    df_SongsFeatures.to_csv('FewSongs.csv')
    return df_SongsFeatures

def addEmbeddingsTodf(df,df_OneSong,col):
    words = df.loc[:,col]
    outputs = (PreprocessDataset.getEmbedding(words[0]))
    for i in range(1,len(words)):
        outputs = np.append(outputs,(PreprocessDataset.getEmbedding(words[i])),axis=0)
    
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

def run(df,df_OneSong,name):
    #Preprocess explicit
    type_map = {False: 0.0, True: 1.0}
    df_OneSong['explicit'] = df_OneSong["explicit"].map(type_map)

    #Preprocess album_type
    type_map = {'single': 0.0, 'album': 1.0, 'compilation': 1.0}
    df_OneSong['album_type'] = df_OneSong["album_type"].map(type_map)

    embedding_features_list = ['album_name','release_date','artist_name','name']

    for col in embedding_features_list:
        df_OneSong = addEmbeddingsTodf(df,df_OneSong,col)
    
    df_OneSong = df_OneSong.drop(['album_name','release_date','artist_name','name'],axis=1)

    df_OneSong.to_csv(name)

if __name__ == "__main__":
    sp = CreateDataset.createSpotifyApiSession('','','')

    track_ids = ['7u0Mn9qAgZxcSWm0db2PaG','3hfmh1XIlJp2Uis4kWboqJ','0DIcssPpatAMqFXLZCxZMN','4n93SK7dQsQVu9BM5QzvAx','0nky5lP13IyrlQEGEzGQZt']
    df_OneSong = getSongsFeatures(sp,track_ids)

    df = pd.read_csv('SongsDataset.csv')

    run(df,df_OneSong,'FewSongsPreprocessed.csv')