from django.urls import path
from . import views

urlpatterns = [
    path('',                         views.home,       name='home'),
    path('analyze/',                 views.analyze,    name='analyze'),
    path('batch/',                   views.batch,      name='batch'),
    path('result/<int:pk>/',         views.result,     name='result'),
    path('history/',                 views.history,    name='history'),
    path('history/delete/<int:pk>/', views.delete,     name='delete'),
    path('stats/',                   views.stats,      name='stats'),
    path('export/',                  views.export_csv, name='export'),
]
