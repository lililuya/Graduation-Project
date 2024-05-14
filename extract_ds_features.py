"""
    Script for extracting DeepSpeech features from audio file.
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from deepspeech_store import get_deepspeech_model_file
from deepspeech_features import conv_audios_to_deepspeech


def extract_features(in_audios,
                     deepspeech_pb_path,
                     config,
                     metainfo_file_path=None):
    """
    Real extract audio from video file.
    Parameters
    ----------
    in_audios : list of str
        Paths to input audio files.
    deepspeech_pb_path : str
        Path to DeepSpeech 0.1.0 frozen model.
    metainfo_file_path : str, default None
        Path to file with meta-information.
    """
    #deepspeech_pb_path="/disk4/keyu/DeepSpeech/deepspeech-0.9.2-models.pbmm"
    if metainfo_file_path is None:
        num_frames_info = [None] * len(in_audios)
    else:
        train_df = pd.read_csv(
            metainfo_file_path,
            sep="\t",
            index_col=False,
            dtype={"Id": np.int, "File": np.unicode, "Count": np.int})
        num_frames_info = train_df["Count"].values
        assert (len(num_frames_info) == len(in_audios))
        
    deep_feature = conv_audios_to_deepspeech(
            audios=in_audios,
            num_frames_info=num_frames_info,
            deepspeech_pb_path=deepspeech_pb_path,
            config = config)
    return deep_feature


"""
把这个代码修改一下，改成返回值是audio
"""
def return_deepfeature(input_audio_path, config, metainfo=None):
    """ Main body of script """
    start_time = time.time()
    in_audio = os.path.expanduser(input_audio_path)
    if not os.path.exists(in_audio):
        raise Exception("Input file/directory doesn't exist: {}".format(in_audio))
    # deepspeech_pb_path = "/mnt/sdb/cxh/liwen/EAT_code/preprocess/DS_model/output_graph.pb"
    deepspeech_pb_path = ""
    if not os.path.exists(deepspeech_pb_path):
        deepspeech_pb_path = get_deepspeech_model_file()
    print(deepspeech_pb_path)
    if os.path.isfile(in_audio):
        feature = extract_features(
                in_audios=[in_audio],
                deepspeech_pb_path = deepspeech_pb_path,
                config = config,
                metainfo_file_path = metainfo)
    else:
        audio_file_paths = []
        for file_name in os.listdir(in_audio):
            if not os.path.isfile(os.path.join(in_audio, file_name)):
                continue
            _, file_ext = os.path.splitext(file_name)
            if file_ext.lower() == ".wav":
                audio_file_path = os.path.join(in_audio, file_name)
                audio_file_paths.append(audio_file_path)
        audio_file_paths = sorted(audio_file_paths)
        feature = extract_features(
                in_audios = audio_file_paths,
                deepspeech_pb_path = deepspeech_pb_path,
                config = config,
                metainfo_file_path = metainfo)
    end_time = time.time()
    print("<=================== Return deepspeech feature ===================>")
    print(f"{end_time - start_time} \n")
    return feature

if __name__=="__main__":
    input_audio_path = "/mnt/sdb/cxh/liwen/EAT_code/audio_temp/tmp.wav"
    feature = return_deepfeature(input_audio_path)
    print(feature[0].shape)
