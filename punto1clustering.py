import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# PUNTO 1: CLUSTERING DE CLIENTES (GUARDAR IMÁGENES)
# Dataset: Mall Customers
# ---------------------------------------------------------

def main():
    print("⬇Descargando dataset 'Mall Customers'...")
    url = "https://raw.githubusercontent.com/sharmaroshan/Clustering-of-Mall-Customers/master/Mall_Customers.csv"
    
    try:
        df = pd.read_csv(url)
        print("Descarga exitosa.")
    except Exception as e:
        print(f"Error crítico: {e}")
        return

    # Limpieza de columnas
    df.rename(columns={
        'CustomerID': 'ID',
        'Genre': 'Genero', 'Gender': 'Genero',
        'Age': 'Edad',
        'Annual Income (k$)': 'Ingreso_Anual',
        'Spending Score (1-100)': 'Puntaje_Gasto'
    }, inplace=True)

    X = df[['Ingreso_Anual', 'Puntaje_Gasto']].values

    print("Datos cargados correctamente.")
    print("-" * 50)

    # -----------------------------------------------------
    # 2. Método del Codo
    # -----------------------------------------------------
    print("Calculando el Codo...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Graficar y GUARDAR el Codo
    plt.figure(figsize=(10,5))
    plt.plot(range(1, 11), wcss, marker='o', color='red')
    plt.title('Método del Codo (El quiebre óptimo es K=5)')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia (WCSS)')
    plt.grid(True)
    
    # CAMBIO AQUÍ: Guardar en vez de mostrar
    plt.savefig('grafico_codo.png')
    print(" Imagen guardada: grafico_codo.png")
    plt.close() # Cierra la figura para liberar memoria

    # -----------------------------------------------------
    # 3. Aplicar K-Means con K=5
    # -----------------------------------------------------
    print("Entrenando K-Means con 5 grupos...")
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
    y_kmeans = kmeans.fit_predict(X)

    # -----------------------------------------------------
    # 4. Visualización Final
    # -----------------------------------------------------
    plt.figure(figsize=(12,8))
    
    colores = ['red', 'blue', 'green', 'cyan', 'magenta']
    for i in range(5):
        plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], 
                    s=100, c=colores[i], label=f'Cluster {i}')

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=300, c='yellow', label='Centroides', edgecolors='black')

    plt.title('Segmentación de Clientes (Mall Customers)')
    plt.xlabel('Ingreso Anual (k$)')
    plt.ylabel('Puntaje de Gasto (1-100)')
    plt.legend()
    
    # CAMBIO AQUÍ: Guardar en vez de mostrar
    plt.savefig('grafico_clusters.png')
    print("Imagen guardada: grafico_clusters.png")
    plt.close()

    print("-" * 50)
    print("PROCESO FINALIZADO.")
    print("Busca los archivos 'grafico_codo.png' y 'grafico_clusters.png' en tu carpeta.")

if __name__ == "__main__":
    main()