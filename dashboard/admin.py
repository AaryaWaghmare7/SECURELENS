from django.contrib import admin
from .models import ImageAnalysis


@admin.register(ImageAnalysis)
class ImageAnalysisAdmin(admin.ModelAdmin):
    list_display = ('id', 'owner', 'prediction', 'confidence', 'uploaded_at')
    list_filter = ('prediction', 'uploaded_at')
    search_fields = ('image', 'owner__username')
