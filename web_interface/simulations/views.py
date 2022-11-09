import uuid
import json
import requests
import socket

from django.shortcuts import render, redirect, HttpResponseRedirect

from rest_framework import status
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Simulation, SimulationConfig, SimulationTimestep, SimulationPlayer, UserAction
from .serializers import SimulationSerializer, SimulationConfigSerializer, SimulationStepSerializer, SimulationUserActionSerializer
from .forms import UserForm

# send socket request to server
HEADER = 128
SERVER = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 5555  # Port to listen on (non-privileged ports are > 1023)
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "DISCONNECT"

def start_page(request):
    form = UserForm()
    context = {
        'form': form,
        'returiningUser': False,
        'coookie': '',
        'user': None,
    }

    if request.method == 'GET':
        if 'swarm_southampton_cookie' in request.COOKIES.keys():
            context['returningUser'] = True
            context['cookie'] = request.COOKIES['swarm_southampton_cookie']
            context['user'] = SimulationPlayer.objects.get(cookie=context['cookie'])
            context['form'] = UserForm(instance=context['user'])

    if request.method == 'POST':
        response = HttpResponseRedirect('/create')

        if 'cookies-accepted' in request.POST.keys():
            form = UserForm(request.POST)
            if form.is_valid():
                player = form.save()

            # generate random cookie id
            cookie = str(uuid.uuid4())

            # save the cookie to the player
            player.cookie = cookie
            player.save()

            # save cookie
            response.set_cookie('swarm_southampton_cookie', cookie)

        if 'cookies-denied' in request.POST.keys():
            form = UserForm(request.POST)
            if form.is_valid():
                player = form.save()

            player.cookie = 'denied_cookie'
            player.save()

        if 'return-user' in request.POST.keys():
            cookie = request.COOKIES['swarm_southampton_cookie']
            user = UserForm(request.POST, instance=SimulationPlayer.objects.get(cookie=cookie))
            if user.is_valid():
                user.save()

        if 'new-user' in request.POST.keys():
            response = HttpResponseRedirect('/')
            response.delete_cookie('swarm_southampton_cookie')

        return response

    return render(request, "simulations/start_page.html", context)


def simulations(request):
    context = {}
    return render(request, "simulations/simulations.html", context)


def create_simulation(request):
    if request.method == 'POST':
        sim = Simulation(
            name=request.POST['sim_name'],
            level=request.POST['level'],
            status="PR"
        )
        sim.save()

        config = SimulationConfig(
            simulation=sim,
            borderPoints=request.POST['borders']
        )
        config.save()

        requests.get("http://localhost:8000/api/v1/simulations/" + str(sim.id) + "/start_simulation/")

        return redirect("/simulations/" + str(sim.id))


    context = {}
    return render(request, "simulations/create_simulation.html", context)


def view_simulation(request, sim_id):
    sim = Simulation.objects.get(pk=sim_id)
    sim_conf = SimulationConfig.objects.get(simulation=sim)

    context = {
        "simulation": sim.to_json(),
        "config": sim_conf.borderPoints,
    }
    return render(request, "simulations/view_simulation.html", context)


# API

class SimulationViewSet(viewsets.ModelViewSet):
    queryset = Simulation.objects.all()
    serializer_class = SimulationSerializer
    http_method_names = ['get', 'post', 'patch']

    @action(detail=True, methods=['get'])
    def config(self, request, pk):

        if request.method == "GET":
            queryset = SimulationConfig.objects.get(simulation=pk)
            serializer = SimulationConfigSerializer(queryset, many=False, context={'request': request})
            return Response(serializer.data)

        return redirect("/api/v1/simulations/" + str(pk))

    @action(detail=True, methods=['get', 'post'])
    def timestep(self, request, pk, timestep=None):

        if timestep is None:
            queryset = SimulationTimestep.objects.all().filter(simulation=pk)
            serializer = SimulationStepSerializer(queryset, many=True, context={'request': request})
            return Response(serializer.data)
        else:
            if request.method == "GET":
                queryset = SimulationTimestep.objects.all().filter(simulation=pk, timestep=timestep)
                serializer = SimulationStepSerializer(queryset, many=True, context={'request': request})
                return Response(serializer.data)

            elif request.method == "POST":
                step = SimulationTimestep(
                    simulation=Simulation.objects.get(pk=pk),
                    config=request.data["config"],
                    timestep=timestep
                )

                try:
                    step.save()
                    queryset = SimulationTimestep.objects.all().filter(sim=pk, timestep=timestep)
                    serializer = SimulationStepSerializer(queryset, many=True, context={'request': request})

                    return Response(serializer.data, status=status.HTTP_201_CREATED)
                except:
                    return redirect("/api/v1/simulations/"+str(pk))

        return redirect("/api/v1/simulations/" + str(pk))

    @action(detail=True, methods=['get'])
    def start_simulation(self, request, pk):
        if request.method == "GET":
            queryset = Simulation.objects.get(pk=pk)
            queryset.status = "RN"
            queryset.save()
            serializer = SimulationSerializer(queryset, many=False, context={'request': request})

            # prepare json data
            data = {
                "id": pk,
                "operation": "start",
                "config":{
                    "drones": queryset.drones,
                    "width": queryset.width,
                    "height": queryset.height,
                }
            }

            headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
            requests.post("http://localhost:5555", data=json.dumps(data), headers=headers)

            return Response(serializer.data)

    @action(detail=True, methods=['POST'])
    def add_action(self, request, pk):
        queryset = Simulation.objects.get(pk=pk)
        serializer = SimulationSerializer(queryset, many=False, context={'request': request})

        if request.method == "POST":

            # prepare json data
            data = {
                "id": str(queryset.id),
                "operation": "update",
                "type": request.data["action"],
                "location": request.data["pos"]
            }

            headers = {'Content-type': 'application/json', 'Accept': 'application/json'}
            requests.post("http://localhost:5555", data=json.dumps(data), headers=headers)

        return Response(serializer.data)

    @action(detail=True, methods=['GET', 'POST'])
    def records(self, request, pk):
        queryset = Simulation.objects.get(pk=pk).actions
        serializer = SimulationUserActionSerializer(queryset, many=True, context={'request': request})

        if request.method == "POST":
            user = None
            if request.data['cookie'] != 'denied_cookie':
                user = SimulationPlayer.objects.get(cookie=request.data['cookie'])

            action = UserAction(
                action=request.data['action'],
                user=user,
                simulation=Simulation.objects.get(pk=request.data['simulation_id']),
                time=request.data['time'],
                timestep=request.data['timestep']
            )
            action.save()

        return Response(serializer.data)
