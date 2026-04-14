from django.urls import path
from . import views

urlpatterns = [
    path('',                         views.home,                 name='home'),
    path('login/',                   views.SecureLensLoginView.as_view(), name='login'),
    path('register/',                views.register,             name='register'),
    path('logout/',                  views.SecureLensLogoutView.as_view(), name='logout'),
    path('analyze/',                 views.analyze,              name='analyze'),
    path('result/<int:pk>/',         views.result,               name='result'),
    path('history/',                 views.history,              name='history'),
    path('history/delete/<int:pk>/', views.delete,  name='delete'),
    path('stats/',                   views.stats,                name='stats'),
]
