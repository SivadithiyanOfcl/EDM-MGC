from django.shortcuts import render
from django.template import RequestContext
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .utils import get_waveform

import os
import mimetypes
import librosa
import librosa.display
import numpy as np
import io
import base64


def index(request):
    return render(request, 'index.html')


def upload_audio(request):
    if request.method == 'POST' and request.FILES.get('audioFile'):
        uploaded_file = request.FILES['audioFile']
        
        # Check file extension to ensure it's an audio file
        allowed_audio_extensions = ['.mp3', '.wav', '.flac', '.ogg']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension not in allowed_audio_extensions:
            return JsonResponse({'error': 'Invalid file format. Only .mp3, .wav, .flac, and .ogg files are allowed.'}, status=400)

        # Save the uploaded audio file to a temporary directory
        temp_dir = os.path.join(settings.BASE_DIR, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        
        
        with open(temp_file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            # Load audio file with librosa
            y, sr = librosa.load(temp_file_path)
            
            #print("Audio is loaded successfully")
            
            waveform = get_waveform(y, sr)
            
            #print("Waveform is successfully created")
            os.remove(temp_file_path)
            
            # Return a success response with waveform image data
            return JsonResponse({'message': 'OK', 'waveform':waveform})
        except Exception as e:
            # Return an error response if processing fails
            return JsonResponse({'error': str(e)}, status=400)
    else:
        # Return an error response if no audio file was uploaded
        return JsonResponse({'error': 'No audio file uploaded'}, status=400)