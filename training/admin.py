from django.contrib import admin

# Register your models here.
from .models import Session, Datapoint

admin.site.register(Session)
admin.site.register(Datapoint)
