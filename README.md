# TESIS2

Esta aplicación de ejemplo demuestra un flujo de trabajo sencillo para un servicio de pronósticos basado en Flask. Se incluye un `Dockerfile` para crear la imagen y un conjunto mínimo de scripts de entrenamiento.

## Requisitos previos

Antes de poder construir o desplegar la aplicación asegúrate de tener:

- **Python 3.10 o superior** para ejecutar el código localmente.
- Una cuenta y proyecto en **Google Cloud Platform (GCP)** con permisos para usar Cloud Build y Cloud Run.
- **Docker** instalado localmente o habilitar **Cloud Build** en tu proyecto de GCP para construir imágenes.
- Instalados los componentes de la CLI de **`gcloud`** y haber ejecutado `gcloud auth login`.

## Ejecución local

1. Instala las dependencias de Python:

   ```bash
   cd my_forecast_app_v1
   pip install -r requirements.txt
   ```

2. Inicia la aplicación de manera local:

   ```bash
   python app.py
   ```

   La aplicación quedará disponible en `http://localhost:8080`.

## Construir la imagen con Docker

Si cuentas con Docker local, desde la carpeta raíz ejecuta:

```bash
docker build -t forecast-app:latest ./my_forecast_app_v1
```

Y para probarla localmente:

```bash
docker run -p 8080:8080 forecast-app:latest
```

## Despliegue en Cloud Run

1. Primero construye y sube la imagen a Container Registry o Artifact Registry usando Cloud Build:

   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/forecast-app ./my_forecast_app_v1
   ```

2. Luego despliega en Cloud Run (ajusta la región y el proyecto):

   ```bash
   gcloud run deploy forecast-app \
       --image gcr.io/PROJECT_ID/forecast-app \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
   ```

Con esto la aplicación quedará disponible en una URL pública proporcionada por Cloud Run.

