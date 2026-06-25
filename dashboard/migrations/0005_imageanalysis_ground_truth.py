from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dashboard', '0004_imageanalysis_analysis_meta'),
    ]

    operations = [
        migrations.AddField(
            model_name='imageanalysis',
            name='ground_truth',
            field=models.CharField(blank=True, max_length=20, null=True),
        ),
    ]
