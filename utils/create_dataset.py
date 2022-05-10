import os
import zipfile
import io
import requests
from requests.auth import HTTPBasicAuth

from individual import * 
      
if (not os.path.exists('./saved_data')):
    os.mkdir('./saved_data')

if __name__ == '__main__':
    #Load Glove Embeddings
    load_glove_embeddings()

    #Create Train Dataset for Audio, Video, and Text
    AudioDataset('train')
    VideoDataset('train')
    TextDataset('train')

    AudioDataset('test')
    VideoDataset('test')
    TextDataset('test')

    AudioDataset('dev')
    VideoDataset('dev')
    TextDataset('dev')