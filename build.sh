#!/usr/bin/env bash
set -o errexit

python -m pip install -r requirements-django.txt
python manage.py collectstatic --noinput
python manage.py migrate
