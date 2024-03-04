import pickle

import librosa
import pandas as pd
import cv2
import os
import pdb
import numpy as np
from scipy import signal

import fnmatch
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing
from moviepy.editor import VideoFileClip
import pickle
import csv

def read_csv(file_path):
    data = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data[row[1]] = {"label":row[0], "start":row[2], "end":row[3], "set":row[4]}
    return data

class Kinetics_dataset(object):
    def __init__(self, path_to_dataset=None, frame_interval=1, frame_kept_per_second=1, audio_fps = 16000, mode="train"):
        self.path_to_video = []
        for root, _, files in os.walk(os.path.join(path_to_dataset, mode)):
            for filename in fnmatch.filter(files, '*.mp4'):
                self.path_to_video.append(os.path.join(root, filename))


        self.annotations = read_csv("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/annotations/test.csv")
        self.annotations.update(read_csv("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/annotations/val.csv"))
        self.annotations.update(read_csv("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/annotations/train.csv"))

        remove_already_processed = True
        if remove_already_processed :
            already_files = os.listdir("/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/Image-01-FPS/{}".format(mode))
            new_files = []
            for i in tqdm(range(len(self.path_to_video))):
                name = self.path_to_video[i].split("/")[-1].split(".")[0] + ".pkl"
                if name not in already_files:
                    new_files.append(self.path_to_video[i])
            self.path_to_video = new_files


        # ann_list = list(self.annotations.keys())
        # count = 0
        # for i in tqdm(range(len(self.path_to_video))):
        #     if self.path_to_video[i][:-18].split("/")[-1] not in ann_list:
        #         print(self.path_to_video[i], end = "  -  ")
        #         count +=1
        #         print(count)

        # for i in range(len(self.path_to_video)):
        #     print(self.path_to_video[i])
        #     print(self.annotations[self.path_to_video[i].split("/")[-1][0:11]])
        self.frame_kept_per_second = frame_kept_per_second
        self.sr = audio_fps

        self.path_to_save = os.path.join(path_to_dataset, 'Image-{:02d}-FPS'.format(self.frame_kept_per_second))
        self.path_to_save = os.path.join(self.path_to_save, mode)
        if not os.path.exists(self.path_to_save):
            os.mkdir(self.path_to_save)

    def extractImage_SE(self):
        # num_cores = multiprocessing.cpu_count() - 1
        num_cores = 70

        a = Parallel(n_jobs=num_cores)(delayed(self.video2frame_update_SE)( video_path=each_video, frame_save_path= os.path.join(self.path_to_save, each_video.split("/")[-1].split(".")[0]),min_save_frame=10) for each_video in tqdm(self.path_to_video))
        # a = [self.video2frame_update_SE( video_path=each_video, frame_save_path= os.path.join(self.path_to_save, each_video.split("/")[-1].split(".")[0]),min_save_frame=10) for each_video in tqdm(self.path_to_video)]

    def video2frame_update_SE(self, video_path, frame_save_path, min_save_frame=3, fps=1):
        try:
            frame_save_path = frame_save_path
            vid = cv2.VideoCapture(video_path)
            ann = self.annotations[video_path.split("/")[-1][0:11]]
            video_fps = round(vid.get(cv2.CAP_PROP_FPS))

            start_t = int(ann["start"])
            end_t = int(ann["end"])

            images = []
            for i in range(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, image = vid.read()
                if not ret:
                    break
                if i % video_fps == 0:
                    images.append(np.expand_dims(image, 0))
            if len(images) == 0: return
            images = np.concatenate(images)

            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None: return
            audio = video_clip.audio.to_soundarray(fps = self.sr)

            data = {"images":images, "audio": audio, "start":start_t, "end":end_t, "label": ann["label"]}

            # if len(images)<min_save_frame:
            #     print("this file has an issue")
            # else:
            with open(frame_save_path+'.pkl', 'wb') as handle:
                pickle.dump(data, handle)

        except Exception as e:
            print("An unexpected error occurred:", str(e))
    #

ks = Kinetics_dataset(path_to_dataset="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/",
                      mode="train", audio_fps = 16000)
ks.extractImage_SE()
ks = Kinetics_dataset(path_to_dataset="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/",
                      mode="test", audio_fps = 16000)
ks.extractImage_SE()
ks = Kinetics_dataset(path_to_dataset="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/Kinetics/",
                      mode="val", audio_fps = 16000)
ks.extractImage_SE()
# ave.extractWav_SE()


