"""
ASGI config for Bob_stream project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.0/howto/deployment/asgi/
"""

import os
# from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
# from channels.security.websocket import AllowedHostsOriginValidator
from django.core.asgi import get_asgi_application
from django.urls import path
from django.urls import re_path
from gpt1 import consumers

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Bob_stream.settings')

application = get_asgi_application()

django_asgi_app = get_asgi_application()


application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter([
    re_path('wss/sc/', consumers.MySyncConsumer.as_asgi())
]),
})