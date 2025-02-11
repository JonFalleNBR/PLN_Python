import pip
import pipwin
import pyaudio
#import speech_recognition as sr

import speech_recognition as sr
print("Biblioteca instalada corretamente!")

#pip install pipwin
#pip install pyaudio

def rec_voz():
    mic = sr.Recognizer()
    with sr.Microphone() as source:
        # Usa funcao para reduzir ruido
        mic.adjust_for_ambient_noise(source)
        print("Fale no microfone: ")
        audio = mic.listen(source)
    try:
        # Usa o reconhecimento de voz
        res = mic.recognize_google(audio, language='pt-BR')
        print("Resultado: " + res)
    except sr.UnkownValueError:
        print("Erro:" + sr.UnkownValueError)
    return res


rec_voz()

