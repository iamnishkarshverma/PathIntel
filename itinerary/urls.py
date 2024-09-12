from django.urls import path
from .views import itinerary_view, signin_view, signout_view, signup_view, transcribe_audio, chat_page, team_view, contact_view, get_available_cities
from . import views

urlpatterns = [
    path('', itinerary_view, name='itinerary'),
    path('signup/',signup_view, name='signup'),
    path('signin/', signin_view, name='signin'),
    path('signout/', signout_view, name='signout'),
    path('team/', team_view, name='team'),
    path('contact/', contact_view, name='contact'),
    path('available-cities/', get_available_cities, name='available-cities'),

    # chatbot
    path('chat/', views.chat_with_user, name='chat_with_user'),
    path('suggested-questions/', views.get_suggested_questions, name='get_suggested_questions'),
    path('transcribe-audio/', transcribe_audio, name='transcribe_audio'),
    path('chatbot/', views.chat_page, name='chat_page'),
    path('check-authentication/', views.check_authentication, name='check_authentication'),

]
