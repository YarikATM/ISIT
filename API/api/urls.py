from django.contrib import admin
from django.urls import path, include
from api.views import DetectView

urlpatterns = [
    path('detect/', DetectView.as_view(), name="detect"),


]