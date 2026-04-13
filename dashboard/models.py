from django.db import models

class ImageAnalysis(models.Model):
    image       = models.ImageField(upload_to='uploads/')
    prediction  = models.CharField(max_length=20, null=True)
    confidence  = models.FloatField(null=True)
    mean_pixel  = models.FloatField(null=True)
    std_pixel   = models.FloatField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    batch_id    = models.CharField(max_length=50, null=True, blank=True)

    def __str__(self):
        return f"{self.image.name} — {self.prediction}"
