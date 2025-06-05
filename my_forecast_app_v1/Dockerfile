# Usa una imagen base oficial de Python optimizada
FROM python:3.10-slim

# Establece la carpeta de trabajo en el contenedor
WORKDIR /app

# Copia los archivos de requirements primero (para aprovechar cache de Docker)
COPY requirements.txt ./

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia todo el código fuente
COPY . .

# Expone el puerto que Cloud Run espera
EXPOSE 8080

# Comando de inicio usando Gunicorn (mejor que python app.py en producción)
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]
