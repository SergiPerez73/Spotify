import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import argparse

def createSpotifyApiSession(token, client_id,client_secret):
    sp = spotipy.Spotify(auth=token,
                        auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                            client_secret=client_secret))
    
    return sp


def getSongsMarkdf(sp):
    offset = 0

    results = []

    while True:
        response = sp.current_user_top_tracks(50,time_range='long_term',offset=offset)

        if len(response['items']) == 0:
            break

        offset = offset + len(response['items'])
        print(offset)
        results.extend(response['items'])

    df_SongsMark = pd.DataFrame(results)
    df_SongsMark = df_SongsMark[['name','id']]

    df_SongsMark = df_SongsMark.reset_index()
    df_SongsMark = df_SongsMark.rename(columns={"index": "mark"})
    df_SongsMark['mark'] = (df_SongsMark.shape[0] - df_SongsMark['mark']) / df_SongsMark.shape[0]

    df_SongsMark.to_csv('SongsMark.csv')

    return df_SongsMark

def getWantedFeatures(track_dict):
    track_features = {key:[track_dict[key]] for key in ['id','duration_ms','explicit','name','popularity']}
    track_features['album_type'] = [track_dict['album']['album_type']]
    track_features['album_name'] = [track_dict['album']['name']]
    track_features['release_date'] = [track_dict['album']['release_date']]
    track_features['artist_name'] = [track_dict['artists'][0]['name']]
    
    return track_features


def getSongsFeatures(sp,df_SongsMark):
    track_features = {}
    offset = 0
    for track_id in df_SongsMark['id']:
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
    df_SongsFeatures.to_csv('SongsFeatures.csv')
    return df_SongsFeatures

def mergeFeaturesMarkSongs():
    df_songs = pd.read_csv('SongsFeatures.csv')
    df_songs = df_songs.drop(['name','Unnamed: 0'],axis=1)

    df_songsMark = pd.read_csv('SongsMark.csv')
    df_songsMark = df_songsMark.drop(['Unnamed: 0'],axis=1)

    df_merged = df_songs.merge(df_songsMark, how='inner', on='id')
    df_merged = df_merged.drop(['id'],axis=1)

    df_merged.to_csv('SongsDataset.csv')

def addArgs(parser):
    parser.add_argument('-client_id', type=str, default='', required=False, help='Client id')
    parser.add_argument('-client_secret', type=str, default='', required=False, help='Client secret id')
    parser.add_argument('-token', type=str, default='', required=False, help='Session token')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create the dataset.')

    addArgs(parser)

    args = parser.parse_args()

    sp = createSpotifyApiSession(args.token,args.client_id,args.client_secret)
    df_SongsMark = getSongsMarkdf(sp)
    df_SongsFeatures = getSongsFeatures(sp,df_SongsMark)
    mergeFeaturesMarkSongs()


