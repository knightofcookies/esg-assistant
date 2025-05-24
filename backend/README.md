# MacOS

```
$ brew services start redis
```

```
$ celery -A app.main.celery_app worker -l info
```

```
$ uvicorn app.main:app --reload --port 8000
```

# Windows
(Run as administrator)
```
docker run -d --name redis -p 6379:6379 redis:latest
```
(Run as administrator)
```
$ celery -A app.main.celery_app worker -l info
```

```
$ uvicorn app.main:app --reload --port 8000
```
