from django import forms
from .models import ImageAnalysis

class ImageUploadForm(forms.ModelForm):
    class Meta:
        model  = ImageAnalysis
        fields = ['image']
