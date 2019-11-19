from django.db import models

# Create your models here.
class Crawl(models.Model):
    NAME=models.CharField(max_length=20,null=True)
    DATE=models.CharField(max_length=20,null=True)
    IMG=models.CharField(max_length=200,null=True)
    CONTENT=models.TextField(null=True)
    LABEL=models.IntegerField(default=4)
