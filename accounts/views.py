from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, load_model 
from pathlib import Path
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
from IPython.display import Image
from tensorflow.keras.layers import Concatenate, Flatten
from tensorflow.keras.applications.densenet import DenseNet169

from .forms import CreateUserForm
from .models import Account

global_prediction = None

# Create your views here.
def loginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('game')
        else:
            messages.info(request, 'Username or password is incorrect')
    context = {}
    return render(request, 'login.html', context)


def create_user_data(user, form):
    Account.objects.create(user=user, school=form.cleaned_data.get('school'), username=form.cleaned_data.get('username'))


def signUpPage(request):
    form = CreateUserForm()

    if request.method == "POST":
        form = CreateUserForm(request.POST)
        if form.is_valid():

            #added
            user = form.save()
            user.refresh_from_db()

            #form.save()
            messages.success(request, "Registration Successful")

            #Account.objects.update_or_create(user=user, school=form.cleaned_data.get('school'))

            #user.account.school = form.cleaned_data.get('school')
            create_user_data(user,form) 

            return redirect('login')
        else:
            messages.error(request, "Invalid Registration Details")
        
    context = {'form': form}
    return render(request, 'register.html', context)


@login_required(login_url='login')
def game(request):
    context = {}
    return render(request, 'game.html', context) 


@login_required(login_url='login')
def predictImage(request):
    base_dir = Path(__file__).resolve().parent.parent
    clf_model = keras.models.load_model(os.path.join(base_dir, 'accounts', 'model', 'concatenated_model_raw.h5'))
    #clf_model = keras.models.load_model('C:/Users/Laven/Desktop/College Stuff/Artificial Intelligence/yass_ai-1/accounts/models/concatenated_model_raw.h5')
    # will store the image in the media folder so that we can have a an image path for the model
    file_object = request.FILES['filePath']
    file_system = FileSystemStorage()
    file_system.save(file_object.name, file_object)

    filename = 'media/' + str(file_object)
    Image(filename,width=160,height=160)
    img = image.load_img(filename,target_size=(160,160))

    # importing pretrained MobileNet
    mnet_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')

    # importing pretrained DenseNet
    conv_base = DenseNet169(include_top=False, weights='imagenet', input_shape=(160,160,3))

    # preprocessing and then using pretrained neural nets to extract features to be fed into Global Pooling
    im_toarray = tf.keras.preprocessing.image.img_to_array(img)
    im_toarray = np.expand_dims(img, axis=0)
    im_toarray = tf.keras.applications.mobilenet_v2.preprocess_input(im_toarray) 

    #extract features using mobilenet and densenet
    img_features1 = mnet_model.predict(im_toarray)
    img_features2 = conv_base.predict(im_toarray)

    #concatenate features
    mobilenet_img_features = Flatten()(img_features1)
    densenet_img_features = Flatten()(img_features2)
    img_features = np.concatenate((densenet_img_features, mobilenet_img_features), axis=1)

    img_features = np.reshape(img_features, (1,1,1,73600))

    predictions = clf_model.predict(img_features)

    classes = np.argmax(predictions, axis = 1)

    #translate prediction value into name of waste category:
    categories = {
        0: 'Biodegradable', 1: 'Cardboard', 2: 'Glass',
        3: 'Metal', 4: 'Paper', 5: 'Plastic',
        6: 'Sanitary'
    }

    predicted_class = ""
    for i in categories:
        if i == classes[0]:
            predicted_class = categories[i]

    p = []
    
    for i in range(0, len(predictions[0])):
        p.append(predictions[0][i])
    
    print(max(p))

    isDetermined = False
    if max(p) >= 0.60:
        isDetermined = True
    
    global global_prediction
    global_prediction = str(predicted_class)

    context = {'prediction': str(predicted_class), 'target': filename, 'isDetermined': isDetermined}
                
    return render(request, 'game.html', context)


@login_required(login_url='login')
def checkAnswer(request):
    global global_prediction

    basis = global_prediction #placeholder
    answer = request.POST['waste_type']
    verdict = ""
    account = Account.objects.get(user= request.user)

    if basis == str(answer):
        verdict = "correct"
        account.score = account.score + 100
        account.save()
    else:
        verdict = "wrong"
        if account.score < 50:
            account.score = 0
            account.save()
        else:
            account.score = account.score - 50
            account.save()
    context={'verdict': verdict, 'basis': basis, 'score':account.score}
    return render(request, 'game.html', context)


@login_required(login_url='login')
def userProfile(request):
    account_list = Account.objects.all()
    account_list_ordered = Account.objects.order_by('-score')
    for index1, i in enumerate(account_list):
        for index2, j in enumerate(account_list_ordered):
            if account_list[index1].username == account_list_ordered[index2].username:
                account_list[index1].ranking = index2 + 1
                account_list[index1].save()
    account = Account.objects.get(user= request.user)
    context = {'account': account, 'score': account.score, 'ranking': account.ranking}
    return render(request, 'user.html', context)

@login_required(login_url='login')
def rankings(request):
    account_list = Account.objects.all()
    account_list_ordered = Account.objects.order_by('-score')
    for index1, i in enumerate(account_list):
        for index2, j in enumerate(account_list_ordered):
            if account_list[index1].username == account_list_ordered[index2].username:
                account_list[index1].ranking = index2 + 1
                account_list[index1].save()
    return render(request, 'rankings.html', {'account_list': account_list_ordered})

@login_required(login_url='login')
def logout_request(request):
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return redirect("login")