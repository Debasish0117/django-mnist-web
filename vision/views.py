import os
from django.shortcuts import render
from django.http import JsonResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from .forms import UploadForm
from ml.infer import load_model_once, infer_image


_model = None


def get_model():
    global _model
    if _model is None:
        ckpt = os.getenv('CKPT_PATH')
        _model = load_model_once(ckpt)
    return _model


def index(request: HttpRequest):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['image']
            model = get_model()
            digit = infer_image(model, img)
            return render(request, 'vision/result.html', {'digit': digit})
    else:
        form = UploadForm()
    return render(request, 'vision/upload.html', {'form': form})


@csrf_exempt
def predict_api(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'image is required'}, status=400)
    model = get_model()
    digit = infer_image(model, request.FILES['image'])
    return JsonResponse({'digit': digit})


