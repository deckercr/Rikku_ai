from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat, name='chat'),
    path('identity/enroll/', views.enroll, name='enroll'),
    path('identity/profiles/', views.profiles, name='profiles'),
    path('identity/profiles/<int:profile_id>/', views.profiles, name='profile-detail'),
    path('identity/identify/', views.identify, name='identify'),
]
