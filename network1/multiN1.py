#!/usr/bin/env python
# coding: utf-8

#import statements
import pandas as pd
import matplotlib.pyplot as plt
import glob
import math
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import re
from sklearn.cluster import KMeans
import numpy as np
import multiprocessing
from multiprocessing import Process, Queue
import time

#get_file_list seraches the data folder and returns a list of input files
def get_file_list():
    #Saving all of the filepaths in data
    data = []
    for folder in glob.glob("../data/*"):
        if (folder[-3:] != '.md' and folder[-6:] != '.ipynb'):
            #print(glob.glob(folder+'/*')[0])
            data.append(glob.glob(folder+'/*')[0])
    return data

data_columns = ['head_x', 'head_y', 'head_z',
               'neck_x', 'neck_y', 'neck_z',
               'spine_x', 'spine_y', 'spine_z',
               'hip_x', 'hip_y', 'hip_z',
               'shoulderl_x', 'shoulderl_y', 'shoulderl_z',
               'shoulderr_x', 'shoulderr_y', 'shoulderr_z',
               'elbowl_x', 'elbowl_y', 'elbowl_z',
               'elbowr_x', 'elbowr_y', 'elbowr_z',
               'wristl_x', 'wristl_y', 'wristl_z',
               'wristr_x', 'wristr_y', 'wristr_z',
               'handl_x', 'handl_y', 'handl_z',
               'handr_x', 'handr_y', 'handr_z',
               'handtipl_x', 'handtipl_y', 'handtipl_z',
               'handtipr_x', 'handtipr_y', 'handtipr_z',
               'hipl_x', 'hipl_y', 'hipl_z',
               'hipr_x', 'hipr_y', 'hipr_z',
               'kneel_x', 'kneel_y', 'kneel_z',
               'kneer_x', 'kneer_y', 'kneer_z',
               'anklel_x', 'anklel_y', 'anklel_z',
               'ankler_x', 'ankler_y', 'ankler_z',
               'footl_x', 'footl_y', 'footl_z',
               'footr_x', 'footr_y', 'footr_z']

#sets up spotify client parameters
def set_spotify():
    client_id = 'd0b2731526744c759fcf012a56ec5bd5'
    client_secret = '6e593cabd0e043da9041c5ef5825dec7'

    #Sets up authentication to use the Spotify API
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    #Creates a Spotipy session using the credentials
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

sp = set_spotify()

#retrieves beat data from spotify API
def get_beats(song_id, filename):
    analysis = sp.audio_analysis(song_id)
    #features = sp.audio_features(song_id)
    #Starting beat will change depending on song
    eight_counts = int(re.search("\d+.txt", filename).group()[0])
    beat_count = eight_counts * 8 #not sure if this should be 4 or eight, we will know when we can visualize
    beats = analysis['beats'][beat_count:]
    return beats

#adds spotify beat data to kinect data
def add_beats(dance, beats, clap_frame, filename):
    eight_counts = int(re.search("\d+.txt", filename).group()[0])
    beat_count = eight_counts * 8 #not sure if this should be 4 or eight, we will know when we can visualize
    #add time stamps to dataframe
    dance.loc[clap_frame, "time_stamp"] = beats[beat_count]["start"]

    current_beat = 0
    #for each row in the data frame...
    for index, row in dance.iterrows():
        time_stamp = beats[0]["start"] + 0.0666666666*(index-clap_frame)
        dance.loc[index, "time_stamp"] = time_stamp #set time stamp for each frame
        if current_beat < len(beats) - 1:
            if beats[current_beat + 1]["start"] < time_stamp:
                current_beat += 1
        dance.loc[index, "beat_index"] = current_beat #set beat index for each frame

    clap_to_end = dance[dance["time_stamp"] < beats[-1]["start"]].copy() #cut off frames where song ends

    return clap_to_end

#parses spotify ID from filename using regex
def extrapolate_id(dataname):
    id_container = re.search("/\w{22}_", dataname).group()
    track_id = id_container[1:-1]
    return track_id

#segment_beats takes the result of add_beats and returns a list of dataframes of individual beats to add to training set
def segment_beats(dance_data):
    #groups the dance data by their beat index
    groups = dance_data.groupby('beat_index')
    #initialize empty list to populate with song beats
    song_beats = []
    #iterate through each group and append to song_beats
    for name, group in groups:
        song_beats.append(group)
    return song_beats

#gets the filename for a txt of raw kinekt data with specified naming convention
#returns a df of beats that can be appended to the training set
def standardize_beats(filename):
    
    ## IMPORTING 3D POINT DANCE DATA FROM KINECT
    
    array2d = []
    
    #opening file
    fp = open(filename, 'r')
    line = fp.readline()
    
    #find first line with all data points accounted for
    #this serves to skip any frames where the square_handtip_distance would be improperly calculated
    frame_min = 0
    #this tracks how many lines are skipped so the df index lines up with the line number of the txt files
    lineno = 1
    while frame_min == 0:
        frame = line.split()
        for i in range(len(frame)):
            frame[i] = abs(float(frame[i]))
        line = fp.readline()
        frame_min = min(frame)
        lineno += 1
        

    #parsing lines of txt file
    while line:
        #splitting numbers into array
        frame = line.split()
        #converting strings to floats
        for i in range(len(frame)):
            frame[i] = float(frame[i])
            #adding frame to array
        array2d.append(frame)
        line = fp.readline()
        #checking to see if all data is missing (implies dancer left screen, df can end) and ending parsing while loop
        if sum(map(abs, frame)) == 0.0:
            line = 0

    #inputting file into dataframe
    df = pd.DataFrame(array2d, columns = data_columns)
    #adjusting df index to line up with txt files
    df.index += lineno

    #defining distance between handtips
    df["square_handtip_distance"] = (df['handtipl_x'] - df['handtipr_x'])**2 + (df['handtipl_y'] - df['handtipr_y'])**2 + (df['handtipl_z'] - df['handtipr_z'])**2
    
    #currently is just an estimation, taking the first frame where the hand distance is less than 0.01
    clap_frame = df[df["square_handtip_distance"] < 0.01].index[0].copy()
    
    #add columns for time stamp and beat index
    df["time_stamp"] = 0
    df["beat_index"] = 0
    dance = df.loc[clap_frame:].copy()

    ## GETTING BEAT INFORMATION FROM SPOTIPY API
    song_id = extrapolate_id(filename)
    beats = get_beats(song_id, filename)
    ## ADDING BEAT INFORMATION TO DANCE DATA AND TRUNCATING
    new_dance = add_beats(dance, beats, clap_frame, filename)
    
    curr_frame = clap_frame
    last_frame = new_dance.index[-1]

    beats_df_start = pd.DataFrame(columns = ["start_time"] + data_columns)
    beats_df_mid = pd.DataFrame(columns = ["start_time"] + data_columns)
    beats_df_end = pd.DataFrame(columns = ["start_time"] + data_columns)

    
    ## STANDARDIZING BEAT FEATURES TO BE SUITABLE FOR TRAINING
    #this section runs very slowly. If possible, we should try to speed it up
    #last beat is not included because it was excluded from the training set
    ##Start of Beat
    for beat in beats[:-1]:
        beat_data = [beat["start"]]
        if (curr_frame < last_frame):
            while new_dance.loc[curr_frame + 1]["time_stamp"] < beat["start"]:
                curr_frame += 1
            if new_dance.loc[curr_frame + 1]["time_stamp"] > beat["start"]:
                #pinpointing position at beat
                #only for head_x right now, needs to be expanded to all points
                for point in data_columns:
                    f1 = new_dance.loc[curr_frame]
                    f2 = new_dance.loc[curr_frame + 1]
                    beat_pos = f1[point] + (beat["start"] - f1.time_stamp) * ((f2[point] - f1[point]) / (f2.time_stamp - f1.time_stamp))
                    #print(beat_pos)
                    beat_data.append(beat_pos)
        #creating a 1 row df for the beat
        beat_line = pd.DataFrame([beat_data], columns = ["start_time"] + data_columns)
        beats_df_start = beats_df_start.append(beat_line)

    curr_frame = clap_frame
    last_frame = new_dance.index[-1]

    #Middle of Beat
    for beat in beats[:-1]:
        beat_data = [beat["start"]]
        beat_mid = beat["start"] + beat["duration"] / 2
        if (curr_frame < last_frame):
            while new_dance.loc[curr_frame + 1]["time_stamp"] < beat_mid:
                curr_frame += 1
            if new_dance.loc[curr_frame + 1]["time_stamp"] > beat_mid:
                #pinpointing position at beat
                #only for head_x right now, needs to be expanded to all points
                for point in data_columns:
                    f1 = new_dance.loc[curr_frame]
                    f2 = new_dance.loc[curr_frame + 1]
                    beat_pos = f1[point] + (beat_mid - f1.time_stamp) * ((f2[point] - f1[point]) / (f2.time_stamp - f1.time_stamp))
                    beat_data.append(beat_pos)
        #creating a 1 row df for the beat
        beat_line = pd.DataFrame([beat_data], columns = ["start_time"] + data_columns)
        beats_df_mid = beats_df_mid.append(beat_line)

    #End of Beat
    curr_frame = clap_frame
    last_frame = new_dance.index[-1]
    for beat in beats[:-2]:
        beat_data = [beat["start"]]
        beat_end = beat["start"] + beat["duration"]
        if (curr_frame < last_frame):
            while new_dance.loc[curr_frame + 1]["time_stamp"] < beat_end:
                curr_frame += 1
            if new_dance.loc[curr_frame + 1]["time_stamp"] > beat_end:
                #pinpointing position at beat
                #only for head_x right now, needs to be expanded to all points
                for point in data_columns:
                    f1 = new_dance.loc[curr_frame]
                    f2 = new_dance.loc[curr_frame + 1]
                    beat_pos = f1[point] + (beat_end - f1.time_stamp) * ((f2[point] - f1[point]) / (f2.time_stamp - f1.time_stamp))
                    beat_data.append(beat_pos)
        #creating a 1 row df for the beat
        beat_line = pd.DataFrame([beat_data], columns = ["start_time"] + data_columns)
        beats_df_end = beats_df_end.append(beat_line)

    beats_df_end = beats_df_end.append(new_dance.loc[last_frame][:66])
    beats_df_end.loc[last_frame, "start_time"] = beats[-2]["start"] #setting this manually since the data doesn't go this far

    mid_and_end = beats_df_mid.merge(beats_df_end, on="start_time", how="outer", suffixes=["", "_1"])
    standard_beats = beats_df_start.merge(mid_and_end, on="start_time", how="outer", suffixes=["_0", "_1/2"])
    
    return standard_beats


def create_training_beats(file_list, q, index):

    #defining columns for beat structure
    start = [name + '_0' for name in data_columns]
    mid = [name + '_1/2' for name in data_columns]
    end = [name + '_1' for name in data_columns]
    #cols should be altered as we do more feature engineering, perhaps adding more snapshots or other data (velocity and such)
    cols = ["start_time"] + start + mid + end
    training_df_of_beats = pd.DataFrame(columns=cols)

    for file in file_list:
        file = file.replace('\\', '/')
        print(file)
        beats = standardize_beats(file)
        training_df_of_beats = pd.concat([training_df_of_beats, beats])
    
    q.put([index, training_df_of_beats])
    print("All data imported in thread {}.".format(index))

def create_k_means(data, q, k):
    print("Testing for k={}.".format(k))
    kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
    kmeans.fit(data)
    q.put([k, kmeans.inertia_]) #multiplying by i here is used to create a minimum as k increases
    
#setting up dataframe for 2nd neural network
def create_move_df(filename):
    song_id = extrapolate_id(filename)
    beats = get_beats(song_id, filename)
    features = sp.audio_features(song_id)
    song_features = features[0]
    analysis = sp.audio_analysis(song_id)
    sections = analysis['sections']
    curr_section_num = 0
    curr_section = sections[curr_section_num]

    #can add whatever features we feel like later, these ones felt like the most important for now
    cols = ["start_time", "duration", "section", "tempo", "danceability", "energy", "valence"]
    training_df_of_moves = pd.DataFrame(columns=cols)

    #add data to data frame (start, duration, section, tempo, features)
    for beat in beats[:-1]:
        #sets section number for current beat
        if (curr_section["start"] + curr_section["duration"]) < beat["start"]:
            if curr_section_num + 1 != len(sections):
                curr_section_num = curr_section_num + 1
            curr_section = sections[curr_section_num]
        
        #creating data in row format for df
        beat_data = [beat["start"], beat["duration"], curr_section_num, curr_section["tempo"], song_features["danceability"], song_features["energy"], song_features["valence"]]
        beat_line = pd.DataFrame([beat_data], columns=cols)
        training_df_of_moves = training_df_of_moves.append(beat_line)
    
    return training_df_of_moves.reset_index(drop=True)


def basic_func(x):
    if x == 0:
        return 'zero'
    elif x%2 == 0:
        return 'even'
    else:
        return 'odd'

def multiprocessing_func(x):
    y = x*x
    time.sleep(1)
    print('{} squared results in a/an {} number'.format(x, basic_func(y)))
    
if __name__ == '__main__':
    print("The algorithm is running! Sit back and enjoy these print statement updates.")

    # DETECTING FILES
    files = get_file_list() #REMEMBER TO REMOVE THIS INDEX
    num_files = len(files)
    print("Detected " + str(num_files) + " files of 3D dance data.")

    # IMPORTING DATA INTO TRAINING SET
    print("Now importing 3D data from files. When a new file is opened, its name will be printed.")
    print("This is a multithreaded process, each process will print when it finishes.")
    num_threads = 8
    processes = [None] * num_threads
    results = [None] * num_threads

    #sets up a queue for process results to be stored in
    q = Queue()

    #creates separate processes to optimize import speed
    for i in range(num_threads):
        p = multiprocessing.Process(target=create_training_beats, args=(files[num_files*(i)//num_threads:num_files*(i+1)//num_threads], q, i))
        processes[i] = p
        p.start()
        
    #pulls results from process queue (needs to be done before joining)
    for process in processes:
        idx, beats = q.get()
        results[idx] = beats

    #ends all processes
    for process in processes:
        process.join()

    training_df_of_beats = results[0]
    #combines results from all the threads
    for df in results[1:]:
        training_df_of_beats = pd.concat([training_df_of_beats, df])

    training_df_of_beats["start_time"] = training_df_of_beats["start_time"] / 300 #cutting this down so start_time doesn't dominate clustering
    #print(training_df_of_beats)
    print("All data imported.")

    # MACHINE LEARNING
    
    print("Now running k-means clustering for different values of k to determine the optimal number of clusters.")
    print("The chosen k will represent how many different dance moves we classify.")
    k_max = 40
    k_range = range(1, k_max + 1)
    wcss = [None] * k_max
    processes = [None] * k_max

    cue = Queue()

    #creates separate processes to optimize import speed
    for i in k_range:
        p = multiprocessing.Process(target=create_k_means, args=(training_df_of_beats, cue, i))
        processes[i - 1] = p
        p.start()
        
    #pulls results from process queue (needs to be done before joining)
    for process in processes:
        k, score = cue.get()
        print(k, score)
        wcss[k-1] = score

    #ends all processes
    for process in processes:
        process.join()

    curve = list(map(abs, (np.diff(wcss,2)).tolist()))

    chosen_k = curve.index(min(curve)) + 1
    print(str(chosen_k) + " has the lowest wcss (within-cluster squared sum) proportionate to itself.")
    print("Dance moves will be classified into " + str(chosen_k) + " groups.")
    kmeans = KMeans(n_clusters = chosen_k, init = 'k-means++', random_state = 42)
    kmeans.fit(training_df_of_beats)
    all_predictions = kmeans.predict(training_df_of_beats)

    # SETTING UP INPUT FOR 2ND STEP
    print("Now obtaining Spotipy data about song features, each file name will be printed as it is accessed.")
    cols = ["start_time", "duration", "section", "tempo", "danceability", "energy", "valence"]
    training_df_of_moves = pd.DataFrame(columns=cols)

    for file in files:
        file = file.replace('\\', '/')
        print(file)
        moves = create_move_df(file)
        training_df_of_moves = pd.concat([training_df_of_moves, moves])
        
    training_df_of_moves
    print("Spotify data successfully obtained for all songs. Now attaching the dance moves to their respective features.")
    training_set = training_df_of_moves.reset_index().merge(pd.DataFrame(data=all_predictions, columns=["move_class"]), left_index=True, right_index=True)
    training_set.to_csv("net2input.csv", index=False)
    print("Data successfully merged, data saved to net2input.csv.")