import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def main():
    print("🏡 INICIANDO SISTEMA DE PREDICCIÓN DE VIVIENDA (Regresión Ridge)...")
    print("-" * 60)

    # ---------------------------------------------------------
    # 1. CARGA DE DATOS
    # ---------------------------------------------------------
    print("⬇️  Cargando dataset 'California Housing' desde sklearn...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    
    print(f"✅ Datos cargados: {df.shape[0]} registros, {df.shape[1]} columnas.")
    print("📊 Primeras 3 filas:")
    print(df.head(3))
    print("-" * 60)

    # ---------------------------------------------------------
    # 2. PREPROCESAMIENTO (Escalado es vital para Ridge)
    # ---------------------------------------------------------
    print("⚙️  Preprocesando datos...")
    
    # Separar variables
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']

    # División 80% Train - 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalado (StandardScaler) - Ridge es sensible a la escala
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # OJO: Solo transform en test, no fit

    print("✅ Datos escalados y divididos correctamente.")

    # ---------------------------------------------------------
    # 3. ENTRENAMIENTO (Ridge Regression)
    # ---------------------------------------------------------
    print("🚀 Entrenando modelo Ridge (con regularización L2)...")
    
    # alpha=1.0 es la fuerza de la regularización
    model_ridge = Ridge(alpha=1.0) 
    model_ridge.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    # 4. EVALUACIÓN
    # ---------------------------------------------------------
    print("📉 Evaluando modelo...")
    y_pred = model_ridge.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n🏆 METRICAS DE RENDIMIENTO:")
    print(f"   -> R² Score: {r2:.4f} (1.0 es perfecto)")
    print(f"   -> RMSE:     {rmse:.4f} (Error promedio en unidades de $100k)")
    print(f"   -> Error aprox en USD: ${rmse * 100000:,.2f}")

    # Gráfica rápida (Se guarda como imagen)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Línea ideal
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.title('Resultados Regresión Ridge')
    plt.savefig('grafico_prediccion_housing.png')
    print("📸 Gráfica guardada: grafico_prediccion_housing.png")

    # ---------------------------------------------------------
    # 5. SISTEMA DE PREDICCIÓN (Función para nuevas casas)
    # ---------------------------------------------------------
    def predecir_precio(datos_dict):
        # Convertir a DataFrame
        nueva_data = pd.DataFrame([datos_dict])
        
        # ESCALAR usando el mismo scaler del entrenamiento
        nueva_data_scaled = scaler.transform(nueva_data)
        
        # Predecir
        pred = model_ridge.predict(nueva_data_scaled)
        valor_usd = pred[0] * 100000
        return valor_usd

    print("-" * 60)
    print("🔎 PRUEBA DE PREDICCIÓN INDIVIDUAL:")
    
    # Ejemplo: Casa promedio en zona de ingresos medios
    casa_ejemplo = {
        'MedInc': 5.0,        # Ingreso medio (en decenas de miles)
        'HouseAge': 20.0,     # Antigüedad
        'AveRooms': 6.0,      # Habitaciones
        'AveBedrms': 1.0,     # Dormitorios
        'Population': 1000.0, # Población
        'AveOccup': 3.0,      # Ocupantes
        'Latitude': 34.0,     # Latitud
        'Longitude': -118.0   # Longitud
    }
    
    precio_estimado = predecir_precio(casa_ejemplo)
    print(f"   Datos Casa: {casa_ejemplo}")
    print(f"   💰 PRECIO ESTIMADO: ${precio_estimado:,.2f} USD")
    
    print("-" * 60)
    print("✅ PROCESO FINALIZADO.")

if __name__ == "__main__":
    main()