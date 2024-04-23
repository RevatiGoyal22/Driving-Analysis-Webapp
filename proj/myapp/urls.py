from django.urls import path
from . import views

urlpatterns = [
    path("", views.real_time_view, name="main"),
    path("real_time/", views.real_time_view, name="real_time"),
    path("result/", views.result_view, name="result")
]
