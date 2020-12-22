from django.contrib import admin

# Register your models here.
from .models import Session, Datapoint, EvaluationResult

admin.site.register(Session)
admin.site.register(Datapoint)
admin.site.register(EvaluationResult)
