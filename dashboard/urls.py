from django.urls import path
from . import views

urlpatterns = [
    path('',                         views.home,                 name='home'),
    path('about/',                  views.about,                name='about'),
    path('features/',               views.features,             name='features'),
    path('research/',               views.research,             name='research'),
    path('login/',                   views.SecureLensLoginView.as_view(), name='login'),
    path('register/',                views.register,             name='register'),
    path('logout/',                  views.SecureLensLogoutView.as_view(), name='logout'),
    path('analyze/',                 views.analyze,              name='analyze'),
    path('result/<int:pk>/',         views.result,               name='result'),
    path('result/<int:pk>/label/',    views.label_analysis,       name='label-analysis'),
    path('result/<int:pk>/report/',   views.export_report,        name='export-report'),
    path('batch/',                   views.batch_analyze,        name='batch'),
    path('compare/',                 views.compare,              name='compare'),
    path('history/',                 views.history,              name='history'),
    path('history/delete/<int:pk>/', views.delete,  name='delete'),
    path('stats/',                   views.stats,                name='stats'),
]
