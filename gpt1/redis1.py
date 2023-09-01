import redis

myHostname = "bob1.redis.cache.windows.net"
myPassword = "vfBKJ3QxM6e0IWHFyXL8hWbWkwUyrmyLzAzCaD9nLIw="

r = redis.StrictRedis(host=myHostname, port=6380,
                      password=myPassword, ssl=True)

result = r.ping()
print("Ping returned : " + str(result))