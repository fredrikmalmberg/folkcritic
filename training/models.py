from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Session(models.Model):
    """Model representing a training session"""
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    name = models.CharField(
        max_length=200,
        help_text="Enter a name for this session"
        )

class Datapoint(models.Model):
    session = models.ForeignKey('Session', on_delete=models.SET_NULL, null=True)
    tune = models.CharField(max_length=1000)
    liked = models.BooleanField()