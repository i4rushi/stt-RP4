#pyspeech imports
from email.mime import audio
import json
import pyaudio
import vosk
import queue
import numpy as np

#ram tracking imports
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
        #vMem = psutil.virtual_memory()
        # ramMB = vMem.used / (1024**2)  #usage for OS + services + app
        # ramPercent = vMem.percent
        ramMB = process.memory_info().rss / (1024**2)  #usage for app only
        print(f"RAM Usage: {ramMB:.2f} MB")
        time.sleep(5)

# def send_to_hardware(letter_queue, stop_event):
#     """Pop letters from the queue every 2 seconds and send to hardware"""
#     while not stop_event.is_set():
#         time.sleep(2)
#         try:
#             letter = letter_queue.get_nowait()
#             print(f"Sending to hardware: {letter}")
#         except queue.Empty:
#             pass

def open_microphone_stream(audio_device, rate, channels, frames_per_buffer):
    stream = audio_device.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
    stream.start_stream()
    return stream

def recognize_speech(stream, recognizer, stop_event, letter_queue):
    print("Listening...")
    last_partial_text = ""

    #latency variables
    chunk_start = None
    result_times = []
    partial_times = []

    try:
        while not stop_event.is_set():
            chunk_start = time.perf_counter() # when chunk starts

            data = stream.read(4096)
            audio = np.frombuffer(data, dtype=np.int16)
            audio = audio[::3]  # downsample by factor of 3
            data = audio.tobytes()

            if recognizer.AcceptWaveform(data):
                result_time = time.perf_counter() - chunk_start
                result_times.append(result_time)

                result = recognizer.Result()
                text = json.loads(result).get('text', '')
                print("You said: " + text)
                
                # Add letters to the queue
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
                    # print("Partial: " + partial_text)
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

    # Initialize the model
    model_path = "vosk-model-small-en-us-0.15"
    model = initialize_model(model_path)

    # Initialize the audio device
    audio_device = initialize_pyaudio()
    rate = 48000 #sampling rate for VOSK
    channels = 1
    frames_per_buffer = 8192
    stream = open_microphone_stream(audio_device, rate, channels, frames_per_buffer)

    #ram printing thread
    ram_thread = threading .Thread(target=print_ram, args=(stop_event,))
    ram_thread.start()

    #hardware sending thread
    #hardware_thread = threading.Thread(target=send_to_hardware, args=(letter_queue, stop_event))
    #hardware_thread.start()

    # Initialize the recognizer
    recognizer = vosk.KaldiRecognizer(model, 16000)
    recognize_speech(stream, recognizer, stop_event, letter_queue)

    #wait for threads to finish
    ram_thread.join()
    #hardware_thread.join()

    # If recognize_speech() is interrupted, close the stream and terminate the audio device
    audio_device.terminate()

if __name__ == "__main__":
    main()