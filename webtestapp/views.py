from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from webtestapp.horse import get_simulation_dict
import json
import numpy as np
import sys
import os
from .forms import MyForm
import re


def home(request):
   return render(request, "webtestapp/home.html")

speed_to_bins = {
   "low":10,
   "medium":20,
   "high":40,
}

speed_to_test_size = {
    "low":0.1,
    "medium":0.3,
    "high":0.3,
}

def heavy_function(features, speed, feature_combinations):
    
    output, columns = get_simulation_dict(n_bins=speed_to_bins[speed], features=features, test_size=speed_to_test_size[speed], feature_combinations=feature_combinations)
    labels = list(output["bet_percent"])
    data = list(output["kaishuuritu"])
    
    
    #print(columns)
    return labels, data

def extract_string(input_string):
    pattern = r'[^\\\\\\n\\r\s&delete&・ ]+'
    result_list = re.findall(pattern, input_string)
    return result_list


def simulation_view(request):
    if request.method == "POST":
        form = MyForm(request.POST)
        if form.is_valid():
            features = {}
            for field_name, field_value in form.cleaned_data.items():
                if field_value:
                    features[field_name] = field_value

            # 生成された特徴量を取得
            generated_features = request.POST.get('generated_features', '').split(',')
            feature_combinations = []
            for generated_feature in generated_features:
                li = [MyForm.display_to_feature[display] for display in extract_string(generated_feature)]
                if len(li) > 0:
                   feature_combinations.append(li)
                    
            labels = []
            data = []
            speed = request.POST.get("speed")
            labels, data = heavy_function(features=features, speed=speed, feature_combinations=feature_combinations)
            
            my_dict = {
                "labels": labels,
                "data": data,
                "form": form,
            }

            print("--------------------------------------")

            return render(request, "webtestapp/simulation_view.html", my_dict)
    else:
        form = MyForm()
        my_dict = {
            "labels": [],
            "data": [],
            "form": form,
        }
        return render(request, "webtestapp/simulation_view.html", my_dict)