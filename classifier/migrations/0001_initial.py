# Generated by Django 2.2.7 on 2019-11-18 12:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Crawl',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('NAME', models.CharField(max_length=20)),
                ('DATE', models.CharField(max_length=20)),
                ('IMG', models.CharField(max_length=200)),
                ('CONTENT', models.TextField()),
                ('LABEL', models.IntegerField(default=4)),
            ],
        ),
    ]
