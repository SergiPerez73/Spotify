# Spotify Songs Recommendation Model

## Overview

* Description
* Motivation
* Code execution
* State of the art
* Code implementation
    * Obtaining data
    * Dataset preprocessing
    * Pytorch deep learning model

## 1. Description

Construct of a Deep Learning Model to recommendate your favourite Spotify Songs using Spotify for developers Web API. This Spotify API brinds us access to a lot of information from songs, albums and artists. Combining this information with the songs someone has listened the most, it is possible to create a dataset of songs with features of them and a score for every song depending on how much a user has listened to it. Then, we construct a Deep Learning Model that can predict a number from 0 to 1 of how much the user will like any song.

## 2. Motivation

Searching for new music to listen can be challenging and a very tedious process. To ease that process, this model can predict how much we will like any song before listening to it.

## 3. Code execution

Before executing the code, we need to have an account on Spotify for Developers. Inside it, we have to go to the Dashboard and create a new app. Once the app has been created, we would be given a Client ID and a Client Secret that we will need to execute to connect to the API from the code.

As we not only want to extract public information, but also information of our most listened songs, we will need to bring access to ourselved to get a Token that will also be used when connecting to the API. To get that token, we can follow the instructions from the Documentation of the official website:

[Spotify API: Access token](https://developer.spotify.com/documentation/web-api/concepts/access-token)

The pipeline of this project consists of 4 Python programs that can be executed in the correct order and indicating the arguments we want with a Shell or  Batch file.

```bash
#For Linux:
./ExecuteAll.sh
```

```bash
#For Windows:
.\ExecuteAll.bat
```

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

where W is a weight matrix. We have one per layer:
* $ W_1 ∈ \mathbb{R}^{3\times 8}  $
* $ W_2 ∈ \mathbb{R}^{3\times 1}  $

$b$ is the bias and we have also one per layer: $b_1$ and $b_2$. $\sigma$ is the sigmoid function.

## 5. Code implementation

As we mentioned, the implementation of the code is separated into 4 Python programs that create the dataset, preprocess the dataset, preprocess an extra dataset to know your coincidence with some specific songs that you select and the Pytorch impelemtation of the model with the neural network.

## Author

Sergi Pérez Escalante