#!/bin/bash

# Define el puerto en el que se ejecutará la aplicación, predeterminado a 5000 si no se proporciona
export PORT=${PORT:-5000}

# Lanza la aplicación usando uvicorn
uvicorn app.main:app --host 0.0.0.0 --port $PORT
