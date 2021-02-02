import pyaudio
import numpy as np

# Common
CHANNELS = 1
FORMAT = BITDEPTH = pyaudio.paFloat32

# Time Specified Prerecorded
## store
RATE_prerecorded_time_specified = 44100
CHUNKSIZE_prerecorded_time_specified = 4410
## noise
RATE_prerecorded_time_specified_noise = 16000
CHUNKSIZE_prerecorded_time_specified_noise = 1600

# Realtime
RATE_realtime = 16000
CHUNKSIZE_realtime = 1600
THRESHOLD_LVL_realtime = 0.04
NP_DATATYPE = np.float32
number_of_secs_to_sleep_real_time = 5
## Noisy sample parameters
noisy_sample_secs_realtime = 3  # The number of frames to read is set to 3 seconds of frames.
noisy_sample_num_frames_realtime = RATE_realtime * noisy_sample_secs_realtime
##
NO_SECS_TO_PREDICT_realtime = 2
counter_threshold_realtime = NO_SECS_TO_PREDICT_realtime * (RATE_realtime / CHUNKSIZE_realtime)

