from django.forms import ModelForm
from django.contrib.auth.forms import UserCreationForm
from django import forms
from django.contrib.auth.models import User
from django.forms import ValidationError


class CreateUserForm(UserCreationForm):
    school = forms.CharField(required=True)

    def clean_email(self):
            email = self.cleaned_data['email']
            if User.objects.filter(email=email).exists():
                raise ValidationError("Email already exists")
            return email
    def clean_username(self):
        username = self.cleaned_data['username']
        if User.objects.filter(username=username).exists():
            raise ValidationError("Username already exists")
        return username
            
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'school', 'email', 'password1', 'password2'] 

        
        def save(self, commit=True):
            user = super(CreateUserForm, self).save(commit=False)
            user.extra_field = self.cleaned_data["extra_field"]
            if commit:
                user.save()
            return user