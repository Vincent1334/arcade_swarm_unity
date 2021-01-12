from django.contrib import admin
from django.urls import path, include
from rest_framework import routers

from simulations.views import SimulationViewSet

router = routers.DefaultRouter()
router.register('simulations', SimulationViewSet)

urlpatterns = [
    path('', include("simulations.urls")),
    path('simulations/', include("simulations.urls")),
    path('admin/', admin.site.urls),

    # api
    path('api/v1/', include(router.urls)),
    path('api/v1/simulations/<str:pk>/config/', SimulationViewSet.as_view({'get': 'config', 'post': 'config'})),
    path('api/v1/simulations/<str:pk>/timestep/<int:timestep>', SimulationViewSet.as_view({'get': 'timestep', 'post': 'timestep'})),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework')),

]
