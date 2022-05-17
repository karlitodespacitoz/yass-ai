from django.urls import path, include, re_path
from . import views

urlpatterns = [
    path('login/', views.loginPage, name="login"),
    path('', views.game, name="game"),
    path('register/', views.signUpPage, name="register"),
    path('user/', views.userProfile, name="user"),
    path("rankings/", views.rankings, name="rankings"),
    path("logout", views.logout_request, name= "logout"),
    re_path('predictImage', views.predictImage, name="predictImage"),
    re_path('checkAnswer', views.checkAnswer, name="checkAnswer")
]