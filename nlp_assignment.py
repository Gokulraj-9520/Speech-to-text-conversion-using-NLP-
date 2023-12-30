import transformers
from transformers import pipeline,AutoModelForSpeechSeq2Seq,AutoProcessor
import torch
import os
from pprint import pprint
from jiwer import wer

torch_device='cuda:0' if torch.cuda.is_available() else 'cpu'

torch_type=torch.float32 if torch.cuda.is_available else torch.float32

model_type='openai/whisper-large-v3'

model = AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3',torch_dtype=torch_type,use_safetensors=True)

model.to(torch_device)

model_processor=AutoProcessor.from_pretrained('openai/whisper-large-v3')

pipelines=pipeline('automatic-speech-recognition',
                   model=model,
                   tokenizer=model_processor.tokenizer,
                   feature_extractor=model_processor.feature_extractor,
                   max_new_tokens=128,
                   chunk_length_s=30,
                   batch_size=16,
                   return_timestamps=True,
                   torch_dtype=torch_type,
                   device=torch_device)

audio_folder_name='common_voice_test'
audio_files=[]
output_text={}
original_text={}
wer_rate={}

def get_text(audio_file_name, transcript_file):
    with open(transcript_file, 'r', encoding='utf-8-sig') as file:
        for line in file:
          parts = line.split('\t')
          if parts[0] == audio_file_name:
            return parts[1].strip()

    return None

transcript_file = 'common_voice_test/trans.txt'

files='result.txt'
header='FileName            OriginalText           OutputText         WER       '
with open (files,'w') as file:
  file.write( header+ '\n')

def get_transcription(filename:str):
    transcription=pipelines(filename)
    return transcription

for file in os.listdir(audio_folder_name):
    if file.endswith('.wav'):
        audio_files.append(os.path.join(audio_folder_name,file))
        filename=os.path.join(audio_folder_name,file)
        print(filename)
        output=get_transcription(filename)
        print(output['text'])
        out_text=output['text']
        output_text[file]=output['text']
        retrieve_text=get_text(file,transcript_file)
        original_text[file]=retrieve_text
        print(retrieve_text)
        wer_rate[file]=wer(retrieve_text,out_text)
        rate=wer(retrieve_text,out_text)
        data=f"{file} {retrieve_text} {out_text} {rate}"
        with open(files,'a') as words:
          words.write(data+'\n')

