from django.urls import path
from . import views

urlpatterns=[
    path('',views.mainp,name='mainp'),
    path('result/',views.sp,name='sp'),
    path('insert/',views.insert_data,name='insert'),
    path('label0/',views.label0,name='label0'),
    path('label1/',views.label1,name='label1'),
    path('label2/',views.label2,name='label2'),
    path('label3/',views.label3,name='label3'),
    path('label4/',views.label4,name='label4')
]
