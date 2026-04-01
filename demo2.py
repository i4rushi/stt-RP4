import json
import pyaudio
import vosk
import queue
import numpy as np
from scipy.signal import butter, sosfilt

import psutil
import os
import threading
import time

def initialize_model(model_path):
    return vosk.Model(model_path)

def initialize_pyaudio():
    return pyaudio.PyAudio()

def print_ram(stop_event):
    process = psutil.Process(os.getpid())
    while not stop_event.is_set():
        ramMB = process.memory_info().rss / (1024**2)
        print(f"RAM Usage: {ramMB:.2f} MB")
        time.sleep(5)

def open_microphone_stream(audio_device, rate, channels, frames_per_buffer):
    stream = audio_device.open(format=pyaudio.paInt16, channels=channels, rate=rate,
                                input=True, input_device_index=1,
                                frames_per_buffer=frames_per_buffer)
    stream.start_stream()
    return stream

def recognize_speech(stream, recognizer, stop_event, letter_queue, sos):  # <-- sos added
    print("Listening...")
    last_partial_text = ""

    chunk_start = None
    result_times = []
    partial_times = []

    try:
        while not stop_event.is_set():
            chunk_start = time.perf_counter()

            data = stream.read(4096)
            audio = np.frombuffer(data, dtype=np.int16)

            # Anti-aliased downsample: low-pass filter THEN decimate
            audio = sosfilt(sos, audio.astype(np.float32) / 32768.0)  # filter in float32 for precision, normalized to [-1, 1]
            audio = audio[::3] * 32768  # decimate 48k -> 16k and scale back to int16 range
            audio = np.clip(audio, -32768, 32767).astype(np.int16)  # clip to valid int16 range
            data = audio.tobytes()

            if recognizer.AcceptWaveform(data):
                result_time = time.perf_counter() - chunk_start
                result_times.append(result_time)

                result = recognizer.Result()
                text = json.loads(result).get('text', '')
                
                print("You said: " + text)

                for letter in text:
                    letter_queue.put(letter)

                last_partial_text = ""
            else:
                partial_start = time.perf_counter()
                partial_result = recognizer.PartialResult()
                partial_latency = time.perf_counter() - partial_start
                partial_times.append(partial_latency)

                partial_text = json.loads(partial_result).get('partial', '')
                if partial_text != last_partial_text:
                    last_partial_text = partial_text

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if result_times:
            avg_result_latency = sum(result_times) / len(result_times)
            print(f"\nAvg Result Latency: {avg_result_latency*1000:.2f}ms")
            print(f"Max Result Latency: {max(result_times)*1000:.2f}ms")

        if partial_times:
            avg_partial_latency = sum(partial_times) / len(partial_times)
            print(f"Avg Partial Latency: {avg_partial_latency*1000:.2f}ms")

        stop_event.set()
        stream.stop_stream()
        stream.close()

def main():
    stop_event = threading.Event()
    letter_queue = queue.Queue()

    model_path = "vosk-model-small-en-us-0.15"
    model = initialize_model(model_path)

    audio_device = initialize_pyaudio()
    rate = 48000
    channels = 1
    frames_per_buffer = 8192
    stream = open_microphone_stream(audio_device, rate, channels, frames_per_buffer)

    # Build anti-aliasing filter once (not per-chunk)
    # Cutoff at 8kHz (Nyquist of 16kHz target), normalized to 48kHz input Nyquist
    nyquist_ratio = (16000 / 2) / (48000 / 2)  # = 0.3333
    sos = butter(N=8, Wn=nyquist_ratio, btype='low', output='sos')

    ram_thread = threading.Thread(target=print_ram, args=(stop_event,))
    ram_thread.start()

    recognizer = vosk.KaldiRecognizer(model, 16000)
    recognize_speech(stream, recognizer, stop_event, letter_queue, sos)  # <-- pass sos

    ram_thread.join()
    audio_device.terminate()

if __name__ == "__main__":
    main()