import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------

class AutoencoderClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderClassifier, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
   
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        out = self.classifier(encoded)
        return out

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim):
        super(CNN_LSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        lstm_input_size = input_dim // 4 

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=64, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(16 * 64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
 
        lstm_out, _ = self.lstm(x)

        x = lstm_out.reshape(lstm_out.size(0), -1)
        out = self.fc(x)
        return out


@st.cache_resource 
def cargar_modelo(nombre_modelo, input_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        if nombre_modelo == "Autoencoder":
            modelo = AutoencoderClassifier(input_dim)
            modelo.load_state_dict(torch.load('./modelo_autoencoder_bancario.pth', map_location=device))
            modelo.to(device)
            modelo.eval()
            return modelo
            
        elif nombre_modelo == "CNN-LSTM":
            modelo = CNN_LSTM(input_dim)
            modelo.load_state_dict(torch.load('./modelo_cnn_lstm_bancario.pth', map_location=device))
            modelo.to(device)
            modelo.eval()
            return modelo
        
    except Exception as e:
        st.error(f"Error al cargar el modelo. ¿Están los archivos .pth en la carpeta? Detalle: {e}")
        return None

def procesar_fechas(df):
    """Transforma la fecha usando Pandas."""
    df_procesado = df.copy()
    
    df_procesado['f_analisis'] = pd.to_datetime(df_procesado['f_analisis'])
    fecha_minima = df_procesado['f_analisis'].min()
    
    df_procesado['dia_semana'] = df_procesado['f_analisis'].dt.dayofweek
    df_procesado['dia_mes'] = df_procesado['f_analisis'].dt.day
    df_procesado['mes'] = df_procesado['f_analisis'].dt.month
    df_procesado['es_quincena'] = df_procesado['f_analisis'].dt.day.isin([15, 30, 31]).astype(int)
    df_procesado['dias_desde_inicio'] = (df_procesado['f_analisis'] - fecha_minima).dt.days
    
    return df_procesado.drop(columns=['f_analisis'])

def hacer_predicciones(modelo, X_numpy):
    """Genera las predicciones con el modelo de PyTorch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_X = torch.tensor(X_numpy, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = modelo(tensor_X).cpu().numpy()
        preds = (probs > 0.5).astype(int).flatten()
    return preds

st.set_page_config(page_title="Predictor Bancario", layout="wide")

st.sidebar.title("Navegación y Configuración ⚙️")


seccion = st.sidebar.radio(
    "Selecciona la sección:", 
    ["1. Predicción (Nuevos Datos)", "2. Validación (Datos Reales)"]
)

st.sidebar.markdown("---")
archivo_cargado = st.sidebar.file_uploader("Sube tu archivo (.csv o .txt)", type=["csv", "txt"])
tamano_muestra = st.sidebar.radio("Tamaño de la muestra:", [100, 200])
opcion_modelo = st.sidebar.selectbox("Selecciona el modelo:", ["Autoencoder", "CNN-LSTM"])

st.title("🏦 Sistema de Predicción Bancaria")

if archivo_cargado is not None:
    try:
        if archivo_cargado.name.endswith('.txt'):
            df_completo = pd.read_csv(archivo_cargado, sep="\t") 
        else:
            df_completo = pd.read_csv(archivo_cargado)
            
        if seccion == "2. Validación (Datos Reales)" and 'var_rta' in df_completo.columns:
            df_completo = df_completo.dropna(subset=['var_rta'])
            
        df_muestra = df_completo.sample(n=tamano_muestra, random_state=42).copy()
        st.success(f"Archivo cargado. {tamano_muestra} filas listas para procesar.")
        
        if st.button("🚀 Ejecutar Modelo", use_container_width=True):
            with st.spinner('Procesando datos y adivinando...'):
                
                es_validacion = 'var_rta' in df_muestra.columns
                
                if es_validacion:
                    y_real = df_muestra['var_rta'].values
                    df_para_predecir = df_muestra.drop(columns=['var_rta'])
                else:
                    df_para_predecir = df_muestra.copy()

                df_listo = procesar_fechas(df_para_predecir)
                
                columnas_ignorar = ['key', 'num_doc', 'obl17']
                columnas_x = [c for c in df_listo.columns if c not in columnas_ignorar]

                X_numpy = df_listo[columnas_x].values
                
                imputer = SimpleImputer(strategy='mean')
                X_numpy = imputer.fit_transform(X_numpy)
                
                scaler = StandardScaler()
                X_numpy = scaler.fit_transform(X_numpy)
                
                # --------------------------------------------------------

                modelo = cargar_modelo(opcion_modelo, input_dim=X_numpy.shape[1])
                
                if modelo is not None:
                    predicciones = hacer_predicciones(modelo, X_numpy)
                    
                    df_muestra['Etiqueta_Predicha'] = predicciones
                    
                    if seccion == "1. Predicción (Nuevos Datos)":
                        st.subheader("📊 Resultados de la Predicción")
                        
                        st.write("#### Vista de los datos predichos")
                        st.dataframe(df_muestra.head(10))
                        
                        st.write("#### Distribución de las predicciones")
                        conteo = df_muestra['Etiqueta_Predicha'].value_counts().reset_index()
                        conteo.columns = ['Etiqueta_Predicha', 'Cantidad']
                        
                        fig, ax = plt.subplots(figsize=(4, 4))
                        ax.pie(conteo['Cantidad'], labels=conteo['Etiqueta_Predicha'], autopct='%1.1f%%', 
                               colors=['#66b3ff','#ff9999'], startangle=90)
                        ax.axis('equal') 
                        st.pyplot(fig)
                    
                    elif seccion == "2. Validación (Datos Reales)":
                        if not es_validacion:
                            st.error("❌ El archivo subido NO contiene la columna 'var_rta'. Sube un archivo válido o cambia a la sección 1.")
                        else:
                            st.subheader("Evaluación del Modelo")
                            exactitud = accuracy_score(y_real, predicciones)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(" Matriz de Confusión")
                                cm = confusion_matrix(y_real, predicciones)
                                fig, ax = plt.subplots(figsize=(4, 4))
                                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                                disp.plot(ax=ax, cmap='Blues')
                                st.pyplot(fig)
                                
                            with col2:
                                st.write(" Resumen")
                                st.metric(label="Precisión de la Muestra (Accuracy)", value=f"{exactitud*100:.1f}%")
                                st.write("Comparativa Real vs Predicción:")
                                st.dataframe(df_muestra[['var_rta', 'Etiqueta_Predicha']].head(10))

    except Exception as e:
        st.error(f"Hubo un error al procesar el archivo: {e}")