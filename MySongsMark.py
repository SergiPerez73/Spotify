import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

sp = spotipy.Spotify(auth="BQAQTT617pXlNDqVBDKli1BMlKJlUWweG7BNuoLefxMmYdpMnIiQUtnlqTUQyjZZA4j0BZV2D6NSLm3cJ2ln6Q5Bko5H0QM-uS6xXp-OOW-1I2HhXClyhjSIZzxdVkevUVFVuc37yu8dpy1aZBOxOoYxmoYLw8UfECIfe_Rx1Yvroj4TAu1FHH4dNRezn3Q50Ch0tXAQfKO32CbtsgE",
                     auth_manager=SpotifyClientCredentials(client_id="16c7c999caa041fe96aa9e01c2abf86d",
                                                           client_secret="4e10f9170b4444f2889bab3497806147"))



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