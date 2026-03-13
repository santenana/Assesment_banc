# 🏦 Sistema de Predicción Bancaria

Este proyecto parte de prueba Tecnica

Prerrequisitos
Antes de comenzar, asegúrate de tener instalado:
* Git

* Python 3.10+ o Conda

* Docker (opcional, para ejecución en contenedores)

Para ejecutar la interfaz hay dos opciones, clonando el repositorio en local 
o bien ejecutandolo desde Docker

Primero abra la terminal de windows alli ejecute el comando 
```bash
cd Downloads
```
O en la carpeta de su preferencia. Una vez dentro de la carpeta deseada ejecute los siguientes comandos:

```bash
git clone https://github.com/santenana/Assesment_banc.git
cd Assesment_banc
```
Aqui ya esta dentro de la carpeta donde se ejecutara todo, para ello primero vamos a crear un pequeño ambiente virtual de la siguiente manera
```bash
python -m venv venv
.\venv\Scripts\activate
```
una vez creado el ambiente en su carpeta se creara una carpeta con el nombre **venv**, posterior a esto vamos a instalar las dependencias (esto puede tomar unos minutos)
```bash
pip install -r requirements.txt
```

Finalmente escribimos en la terminal:
```bash
streamlit run App_predict_bacolombia.py
```

Esto abrira una pestaña nueva en nuestro navegador predeterminado



