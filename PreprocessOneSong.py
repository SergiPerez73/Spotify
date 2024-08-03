import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import PreprocessDataset


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

if __name__ == "__main__":
    sp = createSpotifyApiSession("BQD34URmIFXq6r6mKPNO00ECYyhcTktEGqRcZ1IZIkOYsChpo5lPqqtZJuwgzOSrLpPA8mJO8pEUCqyuj5TnDEfbLm9zs4rlsSHd9fPnb9B9-4owW0oICskIDbI8rkEs5vaa0XBooxN4FlCrHOJs7o-YFZ6LwCn7PurFY8FGGeeNV6pHCPrjpB6JakfivlfrZUOigcw9l5o8APUr6RU"
                                 ,'16c7c999caa041fe96aa9e01c2abf86d','4e10f9170b4444f2889bab3497806147')

    track_ids = ['7u0Mn9qAgZxcSWm0db2PaG','3hfmh1XIlJp2Uis4kWboqJ','0DIcssPpatAMqFXLZCxZMN','4n93SK7dQsQVu9BM5QzvAx','0nky5lP13IyrlQEGEzGQZt']
    df_OneSong = getSongsFeatures(sp,track_ids)

    #Preprocess
    PreprocessDataset.run(df_OneSong,'OneSongPreprocessed.csv')