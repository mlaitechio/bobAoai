from django.urls import path
from .views import my_view
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('',my_view , name='my_view'),
    
    # path('event-stream/', event_stream, name='event-stream'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)