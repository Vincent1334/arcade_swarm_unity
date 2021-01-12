from rest_framework import serializers
from .models import Simulation, SimulationConfig, SimulationTimestep, UserAction


class SimulationSerializer(serializers.HyperlinkedModelSerializer):
    status = serializers.CharField(source='get_status_display')

    class Meta:
        model = Simulation
        fields = ['url', 'id', 'created_at', 'updated_at', 'name', 'status', 'status_label', 'level', 'width', 'height',
                  'drones']


class SimulationConfigSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SimulationConfig
        fields= ["id", "simulation", "borderPoints"]


class SimulationStepSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = SimulationTimestep
        fields= ["id", "timestep", "config"]


class SimulationUserActionSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = UserAction
        fields= ["id", "action", "simulation"]
