import uuid
from django.db import models
from django.utils.translation import gettext_lazy as _


class CommonInfo(models.Model):
    id = models.UUIDField(primary_key=True, editable=False, default=uuid.uuid4)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class SimulationPlayer(CommonInfo):
    cookie = models.CharField(max_length=1000)
    occupation = models.CharField(max_length=100, null=True)


class Simulation(CommonInfo):
    class Status(models.TextChoices):
        DONE = 'DN', _('Done')
        RUNNING = 'RN', _('Running')
        PREPARING = 'PR', _('Preparing')

    name = models.CharField(max_length=200)
    level = models.CharField(max_length=10)
    status = models.CharField(max_length=2, choices=Status.choices, default=Status.PREPARING)
    status_label = models.CharField(max_length=10)
    width = models.IntegerField()
    height = models.IntegerField()
    drones = models.IntegerField()
    score = models.FloatField(default=0)

    @property
    def time_played(self):
        td = self.updated_at - self.created_at
        return '{0}:{1}:{2}'.format(td.seconds//3600, (td.seconds%3600)//60, td.seconds%60)

    def save(self, *args, **kwargs):
        status = {
            "DN": "success",
            "RN": "warning",
            "PR": "secondary",
        }
        self.status_label = status[self.status]

        # could replace this with a dict of some kind
        if self.level == 'easy':
            self.width = 40
            self.height = 40
            self.drones = 100
        elif self.level == 'medium':
            self.width = 40
            self.height = 40
            self.drones = 15
        elif self.level == 'hard':
            self.width = 40
            self.height = 40
            self.drones = 15
        else:
            pass

        super(Simulation, self).save(*args, **kwargs)

    def to_json(self):
        return{
            "id": str(self.id),
            "drones":self.drones,
            "width": self.width,
            "height": self.height,
        }


class SimulationConfig(CommonInfo):
    borderPoints = models.CharField(max_length=200)  # json to hold an array of corners
    simulation = models.ForeignKey(Simulation, related_name='config', on_delete=models.CASCADE)


class SimulationTimestep(CommonInfo):
    simulation = models.ForeignKey(Simulation, on_delete=models.CASCADE)
    config = models.CharField(max_length=1000)
    timestep = models.IntegerField()


class UserAction(CommonInfo):
    action = models.CharField(max_length=100)
    user = models.ForeignKey(SimulationPlayer, related_name='actions', on_delete=models.CASCADE, null=True)
    simulation = models.ForeignKey(Simulation, related_name='actions', on_delete=models.CASCADE)
    timestep = models.CharField(max_length=100)
    time = models.DateTimeField(null=True)
