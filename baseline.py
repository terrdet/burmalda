#!/usr/bin/env python
"""
 @file   baseline.py
 @brief  Baseline code of simple AE-based anomaly detection (PyTorch port).
 @author Based on work by Ryo Tanabe and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2019 Hitachi, Ltd. All right reserved.
"""
import pickle
import os
import sys
import glob

import numpy as np
import librosa
import librosa.feature
import logging

import torch
import torch.nn as nn

__version__ = "1.0.3 (PyTorch port)"

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def save_pickle(filename, save_data):
    logger.info("save_pickle -> {}".format(filename))
    with open(filename, 'wb') as sf:
        pickle.dump(save_data, sf)


def load_pickle(filename):
    logger.info("load_pickle <- {}".format(filename))
    with open(filename, 'rb') as lf:
        return pickle.load(lf)


def file_load(wav_name, mono=False):
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def demux_wav(wav_name, channel=0):
    try:
        multi_channel_data, sr = file_load(wav_name)
        if multi_channel_data.ndim <= 1:
            return sr, multi_channel_data
        return sr, np.array(multi_channel_data)[channel, :]
    except ValueError as msg:
        logger.warning(f'{msg}')


def file_to_vector_array(file_name,
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    dims = n_mels * frames
    sr, y = demux_wav(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
    vectorarray_size = len(log_mel_spectrogram[0, :]) - frames + 1
    if vectorarray_size < 1:
        return np.empty((0, dims), float)
    vectorarray = np.zeros((vectorarray_size, dims), float)
    for t in range(frames):
        vectorarray[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vectorarray_size].T
    return vectorarray


def list_to_vector_array(file_list,
                         msg="calc...",
                         n_mels=64,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):
    dims = n_mels * frames
    if len(file_list) == 0:
        return np.empty((0, dims), float)
    disable_tqdm = sys.stderr is None
    for idx in tqdm(range(len(file_list)), desc=msg, disable=disable_tqdm):
        vector_array = file_to_vector_array(file_list[idx],
                                            n_mels=n_mels, frames=frames,
                                            n_fft=n_fft, hop_length=hop_length, power=power)
        if idx == 0:
            dataset = np.zeros((vector_array.shape[0] * len(file_list), dims), float)
        dataset[vector_array.shape[0] * idx: vector_array.shape[0] * (idx + 1), :] = vector_array
    return dataset


def dataset_generator(target_dir,
                      normal_dir_name="normal",
                      abnormal_dir_name="abnormal",
                      ext="wav"):
    logger.info("target_dir : {}".format(target_dir))
    normal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{normal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                  normal_dir_name=normal_dir_name, ext=ext))))
    normal_labels = np.zeros(len(normal_files))
    if len(normal_files) == 0:
        logger.warning("no_wav_data!! (normal)")
        return [], [], np.array([]), np.array([])

    abnormal_files = sorted(glob.glob(
        os.path.abspath("{dir}/{abnormal_dir_name}/*.{ext}".format(dir=target_dir,
                                                                    abnormal_dir_name=abnormal_dir_name, ext=ext))))
    abnormal_labels = np.ones(len(abnormal_files))
    if len(abnormal_files) == 0:
        logger.warning("no_wav_data!! (abnormal)")
        return [], [], np.array([]), np.array([])

    train_files = normal_files[len(abnormal_files):]
    train_labels = normal_labels[len(abnormal_files):]
    eval_files = np.concatenate((normal_files[:len(abnormal_files)], abnormal_files), axis=0)
    eval_labels = np.concatenate((normal_labels[:len(abnormal_files)], abnormal_labels), axis=0)
    logger.info("train_file num : {num}".format(num=len(train_files)))
    logger.info("eval_file  num : {num}".format(num=len(eval_files)))

    return train_files, train_labels, eval_files, eval_labels


class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, cond_dim=48, bottleneck_size=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.bottleneck_size = bottleneck_size
        enc_in = input_dim + cond_dim
        self.enc_fc = nn.Sequential(
            nn.Linear(enc_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.mean = nn.Linear(64, bottleneck_size)
        self.log_var = nn.Linear(64, bottleneck_size)
        dec_in = bottleneck_size + cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )
        self.classifier = nn.Linear(64, cond_dim)

    def encode(self, x, cond):
        h = self.enc_fc(torch.cat([x, cond], dim=1))
        return self.mean(h), self.log_var(h), h

    def reparameterize(self, mu, log_var):
        return mu + torch.randn_like(log_var) * torch.exp(0.5 * log_var)

    def decode(self, z, cond):
        return self.decoder(torch.cat([z, cond], dim=1))

    def forward(self, x, cond):
        mu, log_var, h = self.encode(x, cond)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z, cond)
        logits = self.classifier(h)
        return recon, mu, log_var, logits

    @staticmethod
    def kl_loss(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

    @staticmethod
    def classifier_loss(logits, targets):
        return nn.functional.cross_entropy(logits, targets)


def cond_key_to_idx(key, known_keys):
    return known_keys.index(key) if key in known_keys else 0


def make_onehot(idx, dim):
    v = torch.zeros(dim)
    v[idx] = 1.0
    return v
