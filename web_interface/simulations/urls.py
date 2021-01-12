from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.start_page, name='start_page'),
    # path('simulations', views.simulations, name='simulations'),
    path('create', views.create_simulation, name='create_simulation'),
    path('<uuid:sim_id>', views.view_simulation, name='view_simulation'),
]