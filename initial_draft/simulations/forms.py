from django import forms
from .models import SimulationPlayer


class UserForm(forms.ModelForm):
    class Meta:
        model = SimulationPlayer
        fields = ['occupation']
        widgets: {
            'occupation': forms.TextInput(attrs={'class': 'validate', 'id': 'occupation'})
        }
