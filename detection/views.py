from django.shortcuts import render
from .forms import UploadVideoForm
from .utils import predict_video
import tempfile

def upload_video(request):
    result = None
    if request.method == 'POST':
        form = UploadVideoForm(request.POST, request.FILES)
        if form.is_valid():
            video_file = request.FILES['video']
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                for chunk in video_file.chunks():
                    temp_video.write(chunk)
                temp_path = temp_video.name

            label, score = predict_video(temp_path)
            result = f"{label} (Confidence: {score:.2f}%)"
    else:
        form = UploadVideoForm()
    return render(request, 'upload.html', {'form': form, 'result': result})