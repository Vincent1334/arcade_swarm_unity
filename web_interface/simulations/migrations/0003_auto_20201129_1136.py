# Generated by Django 3.1.3 on 2020-11-29 11:36

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('simulations', '0002_useraction'),
    ]

    operations = [
        migrations.AddField(
            model_name='simulation',
            name='drones',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='simulation',
            name='height',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='simulation',
            name='status_label',
            field=models.CharField(default='', max_length=10),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='simulation',
            name='width',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]