# import speech_recognition as sr
#
# from pydub import AudioSegment
# ffmpeg_path  = "C:\\Users\\87290\\Downloads\\ffmpeg-master-latest-win64-gpl\\ffmpeg-master-latest-win64-gpl\\bin\\ffmpeg.exe"
# AudioSegment.converter = ffmpeg_path
# # AudioSegment.ffmpeg("C:/Users/87290/Downloads/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe")
# print(AudioSegment.ffmpeg)w
#
# song = AudioSegment.from_wav("recored_audio_2.wav")
# song.export("recored_audio_2.flac", format="flac")
#
# r = sr.Recognizer()
# with sr.WavFile("./recored_audio_2.wav") as source:
#     audio = r.record(source)  # read the entire audio file
#
# try:
#     print(r.recognize_vosk(audio))
# except sr.UnknownValueError:
#     print("Could not understand audio")
import wave
#
from scipy.io.wavfile import read


import librosa
import soundfile as sf
x,_ = librosa.load('./recored_audio_1.WAV', sr=16000)
sf.write('tmp.wav', x, 16000)
wave.open('tmp.wav','r')

a = read("tmp.wav")
print(a[1])