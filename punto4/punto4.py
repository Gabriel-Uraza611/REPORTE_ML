import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("🚢 INICIANDO CLASIFICADOR DEL TITANIC (Árboles de Decisión)...")
    print("-" * 60)

    # 1. CARGA DE DATOS (Link directo, sin login de Kaggle)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("⬇️  Descargando dataset Titanic...")
    try:
        df = pd.read_csv(url)
        print(f"✅ Datos cargados: {df.shape[0]} pasajeros.")
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        return

    # 2. PREPROCESAMIENTO (Limpieza básica)
    print("🧹 Limpiando datos...")
    
    # Seleccionamos variables útiles (Descartamos Nombre, Ticket y Cabina por complejidad)
    cols_to_use = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    df = df[cols_to_use]

    # Tratamiento de Nulos
    # Rellenamos Edad faltante con la Mediana (lo estándar)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    # Eliminamos las 2 filas sin puerto de embarque
    df.dropna(subset=['Embarked'], inplace=True)

    # 3. CODIFICACIÓN VARIABLES CATEGÓRICAS (Requisito del profesor)
    print("🔢 Convirtiendo variables categóricas (One-Hot Encoding)...")
    # Convertimos 'Sex' y 'Embarked' en números
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    
    print("📊 Variables finales:")
    print(df.columns.tolist())

    # 4. SPLIT DE DATOS
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. AJUSTE DE HIPERPARÁMETROS (GridSearchCV)
    print("⚙️  Buscando los mejores hiperparámetros...")
    
    # Definimos la rejilla de parámetros a probar
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],          # Profundidad máxima del árbol
        'min_samples_split': [2, 5, 10],           # Mínimo de datos para dividir un nodo
        'min_samples_leaf': [1, 2, 4],             # Mínimo de datos en una hoja final
        'criterion': ['gini', 'entropy']           # Criterio de pureza
    }

    # Usamos GridSearch para probar todas las combinaciones
    grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print(f"✅ Mejores parámetros encontrados: {grid.best_params_}")
    print(f"✅ Mejor Accuracy en validación: {grid.best_score_:.2%}")

    # 6. EVALUACIÓN FINAL
    print("📉 Evaluando en Test Set...")
    y_pred = best_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"🏆 EXACTITUD FINAL (Test): {acc:.2%}")
    print("\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # 7. VISUALIZACIÓN
    plt.figure(figsize=(12, 8))
    plot_tree(best_model, feature_names=X.columns, class_names=['Murió', 'Sobrevivió'], filled=True, max_depth=3, fontsize=10)
    plt.title("Árbol de Decisión (Primeros 3 niveles)")
    plt.savefig('grafico_arbol_titanic.png')
    print("📸 Árbol guardado como 'grafico_arbol_titanic.png'")

    print("-" * 60)
    print("✅ PROCESO FINALIZADO.")

if __name__ == "__main__":
    main()