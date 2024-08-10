# Spotify Songs Recommendation Model

## Overview

* Description
* Motivation
* Code execution
* State of the art
* Code implementation
* Author

## 1. Description

Construct of a Deep Learning Model to recommendate your favourite Spotify Songs using Spotify for developers Web API. This Spotify API brinds us access to a lot of information from songs, albums and artists. 

Combining this information with the songs someone has listened the most, it is possible to create a dataset of songs with features of them and a score for every song depending on how much a user has listened to it. Then, we construct a Deep Learning Model that can predict a number from 0 to 1 of how much the user will like any song.

## 2. Motivation

Searching for new music to listen can be challenging and a very tedious process. To ease that process, this model can predict how much we will like any song before listening to it.

## 3. Code execution

### 3.1. Spotify API credentials

Before executing the code, we need to have an account on Spotify for Developers. Inside it, we have to go to the Dashboard and create a new app. Once the app has been created, we would be given a Client ID and a Client Secret that we will need to execute to connect to the API from the code.

As we not only want to extract public information, but also information of our most listened songs, we will need to bring access to ourselved to get a Token that will also be used when connecting to the API. To get that token, we can follow the instructions from the Documentation of the official website:

[Spotify API: Access token](https://developer.spotify.com/documentation/web-api/concepts/access-token)

### 3.2. Check specific songs

Before executing the code, we are able to fill a `FewSongs.csv` with the tracks that we specifically want to know how much we will like (you can also include tracks you already listened to). 

To fill that .csv, you have to indicate the id of the song and the name (the name is only important because it will appear once we execute the code next to the predicted score). The id of the song can be extracted from Spotify on the web as each link of a track has the id of the track.

Here we have an example of the `FewSongs.csv`

|     | id                      | name                            |
|-----|-------------------------|---------------------------------|
| 0   | 3hfmh1XIlJp2Uis4kWboqJ  | Blackeyed Blonde                |
| 1   | 0DIcssPpatAMqFXLZCxZMN  | Antichrist                      |
| 2   | 4n93SK7dQsQVu9BM5QzvAx  | I Can Do It With a Broken Heart |

### 3.3. Execution scripts

The pipeline of this project consists of 4 Python programs that can be executed in the correct order and indicating the arguments we want with a Shell or  Batch file.

```bash
#For Linux:
./ExecuteAll.sh
```

```bash
#For Windows:
.\ExecuteAll.bat
```
On both scripts we have some arguments that we can change some arguments:

* TOKEN, CLIENT_ID and CLIENT_SECRET are the Spotify API credentials that we need to obtain the necessary data.

* TEST_SIZE is the proportion of the subset used to test from 0 to 1. Therefore, this also modifies the size of the training subset.

* LR is the learning rate of the model.

* N_EPOCHS is the number of epochs that we are going to train the data. I recommend a number between 1000 and 3000, but you can try different numbers.

* PRINT_FREQ is the number of epochs that will pass between two prints where it will say the loss of the first batch of the train subset on that epoch.

* TEST_FREQ is the number of epochs that will pass between two tests that will save information on plots that will be shown and saved.

* MODEL_PATH is the path of the model that we want to load, empty if we don't want to load any model.

* N_BATCHES is the number of batches in which the train subset will be divided.

Each execution will save some information from the metrics obtained through the training. Each iteration that a test is done, it is analyzed the accuracy and the loss from the full training and test subset. 

This metrics will be shown as plots at the end of the training and will also be saved to be visualized with tensorboard on the `run_ssrm_pt` folder.

## 4. State of the art

### 4.1. Introduction 

Recommendation models are currently used by a lot of companies to know which content they have to show to each user. Neural networks can be used to approach this problem with two primary methods:

* Collaborative filtering makes predictions based on the past behaviours of users obtaining which products similar users liked.

* Content-Based filtering analyzes features of the products matching them to user's preferences.

Although Recommendation Systems have evolved to use Collaborative filtering, it needs a lot of information of which products likes each user. On our case, we don't have access to such information as we can only know which songs liked a user that has given us permission to extract its prefrences. 

On the other hand, we can create a model that uses Content-Based filtering as we can obtain which songs we have listened to (ordered by the number of times we have listened them) and a lot of features from them thanks to Spotify API. Therefore, it has been decided that the best option for us is Content-Based filtering, which will also be much faster to train.

### 4.2. Features

To train a Content-based model, we need some features from each song that describe it. This will make possible for the neural network to infer if we would like or not the song. More precisely, from each song we will obtain 4 dense features and 4 sparse features.

* Dense features have numerical values. On our case, we will know:
    * The duration of the song in milliseconds
    * If it has or not explicit lyrics (1 or 0)
    * The popularity of the song from 0 to 100
    * The type of the album (single or form an album, 0 or 1)

* Sparse features are categorical features that will need to be transformed into embeddings:
    * Name of the song
    * Name of the album (it will be the name of the song if it's not inside an album)
    * Realease date
    * Name of the artist

To treat dense features is very simple, as we already have numerical values for them. To treat sparse features we will obtain an embedding from each string that represents a category of a categorical feature. This means that we will obtain a numerical vector for them. This numerical feature will have 3 dimensions, that represent the position of each string of the feature in an abstract space that will have in closer positions similar strings.

We also have the score of each song that the model will try to predict. The score will be 1 for our most listened song and 0 for the one that we have listened the least among the songs that we have listened.

### 4.3. Design of the model

The neural network will be very simple (although it can be changed manually if it is considered necessary) as we won't commonly have a lot of data of each user. It will have two layers: the first one will have 3 nodes and the second one 1 node (that will be the output). The relations between layers will be a linear regression passed through a sigmoid function. As we need to predict a number between 0 and 1, this is a useful function for us, as it always outputs a number between 0 and 1, and commonly used for predictions.

In a more formal way:

$$
y = \sigma(W_2\sigma(W_1 x + b_1) + b_2)
$$

where $W$ is a weight matrix. We have one per layer:
* $W_1 ∈ \mathbb{R}^{3\times 8}$
* $W_2 ∈ \mathbb{R}^{3\times 1}$

$b$ is the bias and we have also one per layer: $b_1$ and $b_2$. $\sigma$ is the sigmoid function.

## 5. Code implementation

As we mentioned, the implementation of the code is separated into 4 Python programs that create the dataset, preprocess the dataset, preprocess an extra dataset to know your coincidence with some specific songs that you select and the Pytorch impelemtation of the model with the neural network.

* `CreateDataset.py` is the Python program that obtains the Spotify songs that you have listened the most and gets all necessary features from them.

* `PreprocessDataset.py` converts all features to numerical features. Sparse features are transformed into embeddings and then we apply a dimensionality reduction to have only 3 dimensions per sparse feature.

* `PreprocessFewSongs.py` obtains all the features from the specified features and preprocesses them the same way it is done on the `PreprocessDataset.py`. This allows the following program to do inference with this songs.

* `SSRM_pytorch.py` is the program were we apply a standard scaling to all features and we create Pytorch tensors from the to fed the dataset, separated in multiple batches into the training loop. Then, some metrics from the execution and the score predicted for the tracks from `FewSongs.csv` are shown.

## Author

Sergi Pérez Escalante