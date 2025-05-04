import whisper

if __name__ == '__main__':
    print("start")

    '''
    模型下载地址 在__init__.py _MODELS
    '''
    model = whisper.load_model("D:/Download/ABDM/large-v3-turbo.pt")
    result = model.transcribe("D:/Download/ABDM/Music/16k16bit.mp3", fp16=False)
    print(result["text"])