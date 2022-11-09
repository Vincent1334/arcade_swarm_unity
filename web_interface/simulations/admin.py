from django.contrib import admin
from .models import *


admin.site.register(Simulation)
admin.site.register(SimulationPlayer)
admin.site.register(SimulationConfig)
admin.site.register(SimulationTimestep)
admin.site.register(UserAction)

