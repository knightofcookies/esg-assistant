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