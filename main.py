import whisper

if __name__ == '__main__':
    print("start")

    '''
    模型下载地址 在__init__.py _MODELS
    '''
    model = whisper.load_model("D:/Download/ABDM/large-v3-turbo.pt")
    # result = model.transcribe("16k16bit.mp3", fp16=False)
    # print(result["text"])

    audio = whisper.load_audio("16k16bit.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)

