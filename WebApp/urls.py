from django.urls import path
from . import views

urlpatterns = [
    # AUTH PAGES
    path('', views.registerPage, name='register'),
    path('login/', views.loginPage, name='login'),
    path('otp/', views.otpPage, name='otp'),

    # DASHBOARD / UPLOAD / PREDICT
    path('upload/', views.Upload, name='upload'),
    path('action/upload/', views.UploadAction),
    path('run/existing/', views.RunExisting),
    path('run/propose/', views.RunPropose),
    path('run/lstm/', views.RunLSTM),
    path('graph/', views.Graph),
    path('predict/', views.Predict, name='predict'),
    path('action/predict/', views.PredictAction),
]
