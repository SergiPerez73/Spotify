import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

sp = spotipy.Spotify(auth="your-token",
                     auth_manager=SpotifyClientCredentials(client_id="client-id",
                                                           client_secret="client-secret"))



offset = 0

results = []

while True:
    response = sp.current_user_top_tracks(50,time_range='long_term',offset=offset)

    if len(response['items']) == 0:
        break

    offset = offset + len(response['items'])
    print(offset)
    results.extend(response['items'])

df = pd.DataFrame(results)
df = df[['name','id']]

df = df.reset_index()
df = df.rename(columns={"index": "mark"})
df['mark'] = (1307 - df['mark']) / 1307

df.to_csv('SongsMark.csv')