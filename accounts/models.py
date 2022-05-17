from django.db import models
from django.contrib.auth.models import User

class Account(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    username = models.CharField(max_length=150, unique=True, null=True)
    # first_name = models.CharField(max_length=150, null=True)
    #last_name = models.CharField(max_length=150)
    school = models.CharField(max_length=300)
    #email = models.EmailField(gettext_lazy('email'), unique=True)
    score = models.IntegerField(default=0)
    ranking = models.IntegerField(default=0)