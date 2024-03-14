from django.urls import path
from .import views

app_name = "webtestapp"

urlpatterns = [
  path("", views.home, name="home"),
  path("simulation_view", views.simulation_view, name="simulation_view"),
]