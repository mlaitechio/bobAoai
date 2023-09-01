from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path('^wss/sc/$', consumers.MySyncConsumer.as_asgi()),
]
