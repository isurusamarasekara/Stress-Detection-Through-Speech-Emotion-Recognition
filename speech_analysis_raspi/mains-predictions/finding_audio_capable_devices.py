import pyaudio


def find_devices():
    print('======================================Find the audio capable devices in Raspberry pi======================================')
    p = pyaudio.PyAudio()

    print('======================================All the available audio-capable devices======================================')
    for i in range(p.get_device_count()):
        print(p.get_device_info_by_index(i).get('name'))

    print('======================================Specific audio-capable device======================================')
    for index in range(p.get_device_count()):
        desc = p.get_device_info_by_index(index)
        if desc["name"] == "mic":
            print("DEVICE: %s  INDEX:  %s  RATE:  %s " % (desc["name"], index, int(desc["defaultSampleRate"])))


find_devices()
