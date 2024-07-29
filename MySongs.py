import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="client-id",
                                                           client_secret="client-secret"))

pl_id = 'spotify:playlist:1BKF8qS6E9Ang6BPSagJyT'

offset = 0
results = []

while True:
    response = sp.playlist_items(pl_id,
                                 offset=offset,
                                 fields='items.track.id,total',
                                 additional_types=['track'])

    if len(response['items']) == 0:
        break

    offset = offset + len(response['items'])
    results.extend(response['items'])

def getWantedFeatures(track_dict):
    track_features = {key:[track_dict[key]] for key in ['id','duration_ms','explicit','name','popularity']}
    track_features['album_type'] = [track_dict['album']['album_type']]
    track_features['album_name'] = [track_dict['album']['name']]
    track_features['release_date'] = [track_dict['album']['release_date']]
    track_features['artist_name'] = [track_dict['artists'][0]['name']]
    
    return track_features

track_features = {}
offset = 0
for track_result in results:
    if track_result['track'] is not None:
        track_id = track_result['track']['id']
        track_dict = sp.track(track_id)
        if offset == 0:
            track_features = getWantedFeatures(track_dict)
        else:
            track_features_aux = getWantedFeatures(track_dict)
            for key in track_features:
                track_features[key] = track_features[key] + track_features_aux[key]

        offset +=1
        print(offset)
    
df = pd.DataFrame(track_features)

df.to_csv('Songs.csv')
print('finished')