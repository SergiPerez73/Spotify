import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import umap

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def getEmbedding(text):
    input = tokenizer(text, return_tensors='pt')
    output = model(input['input_ids'])
    return output.last_hidden_state.mean(dim=1).detach().numpy()

def addEmbeddingsTodf(df,col):
    words = df.loc[:,col]
    outputs = (getEmbedding(words[0]))

    for i in range(1,len(words)):
        outputs = np.append(outputs,(getEmbedding(words[i])),axis=0)
    
    umap_model = umap.UMAP(n_components=3, random_state=42)
    embeddings_2d = umap_model.fit_transform(outputs)

    emb_df = pd.DataFrame(embeddings_2d, columns=[col+'1', col+'2', col+'3'])
    df = pd.concat([df, emb_df], axis=1)
    
    print('Finished ',col)

    return df

def run(df,name):

    #Preprocess explicit
    type_map = {False: 0.0, True: 1.0}
    df['explicit'] = df["explicit"].map(type_map)

    #Preprocess album_type
    type_map = {'single': 0.0, 'album': 1.0, 'compilation': 1.0}
    df['album_type'] = df["album_type"].map(type_map)

    embedding_features_list = ['album_name','release_date','artist_name','name']

    for col in embedding_features_list:
        df = addEmbeddingsTodf(df,col)

    df = df.drop(['album_name','release_date','artist_name','name'],axis=1)

    df.to_csv(name)

if __name__ == "__main__":
    df = pd.read_csv('SongsDataset.csv')
    df = df.drop(['Unnamed: 0'],axis=1)

    run(df,'PreprocessedDataset.csv')