import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import warnings

# Ignorar advertencias
warnings.filterwarnings('ignore')

def main():
    print("🛒 INICIANDO SISTEMA DE RECOMENDACIÓN (Fuente Oficial UCI)...")
    print("-" * 60)

    # ---------------------------------------------------------
    # 1. CARGA DE DATOS (Formato Excel Original)
    # ---------------------------------------------------------
    print("⬇️  Descargando dataset 'Online Retail.xlsx' desde UCI Archive...")
    print("    (Esto puede tardar unos segundos porque el archivo pesa 23MB)...")
    
    # URL Oficial de la Universidad de California (UCI) - Esta no se cae.
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    
    try:
        # Usamos engine='openpyxl' para leer el archivo Excel moderno
        df = pd.read_excel(url, engine='openpyxl')
        print(f"✅ Datos cargados: {df.shape[0]} registros.")
    except Exception as e:
        print(f"❌ Error al descargar: {e}")
        print("⚠️ Asegúrate de haber instalado openpyxl: pip install openpyxl")
        return

    # ---------------------------------------------------------
    # 2. LIMPIEZA DE DATOS
    # ---------------------------------------------------------
    print("🧹 Limpiando y preparando datos...")
    
    # Limpiar espacios en descripciones
    df['Description'] = df['Description'].str.strip()
    
    # Eliminar filas sin número de factura
    df.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    
    # Eliminar devoluciones (Empiezan con 'C')
    df = df[~df['InvoiceNo'].str.contains('C')]
    
    # FILTRO CLAVE: Usamos solo "France" para que el proceso sea rápido en tu PC
    print("🌍 Filtrando transacciones de Francia para optimizar memoria...")
    basket = (df[df['Country'] == "France"]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))

    # ---------------------------------------------------------
    # 3. CODIFICACIÓN (One-Hot)
    # ---------------------------------------------------------
    print("🔢 Codificando cestas de compra...")

    def encode_units(x):
        if x <= 0: return 0
        if x >= 1: return 1

    basket_sets = basket.applymap(encode_units)
    
    # Eliminar 'POSTAGE' si existe (es el costo de envío, no un producto)
    if 'POSTAGE' in basket_sets.columns:
        basket_sets.drop('POSTAGE', inplace=True, axis=1)

    print(f"📊 Matriz lista para análisis: {basket_sets.shape[0]} facturas.")

    # ---------------------------------------------------------
    # 4. ALGORITMO APRIORI (Reglas de Asociación)
    # ---------------------------------------------------------
    print("🧠 Buscando patrones frecuentes (Apriori)...")
    
    # min_support=0.07: El producto debe aparecer en el 7% de las compras
    frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)
    
    print(f"✅ Se encontraron {len(frequent_itemsets)} grupos de productos frecuentes.")

    # ---------------------------------------------------------
    # 5. GENERAR REGLAS Y RECOMENDAR
    # ---------------------------------------------------------
    print("🔗 Generando reglas de recomendación...")
    
    # Buscamos relaciones fuertes (Lift >= 1)
    reglas = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    
    # Ordenamos por confianza
    reglas = reglas.sort_values(by='confidence', ascending=False)
    
    # --- FUNCIÓN DE RECOMENDACIÓN ---
    def recomendar(producto):
        print(f"\n🔎 ANÁLISIS: Cliente comprando -> '{producto}'")
        
        # Buscar reglas donde este producto sea el antecedente
        # Usamos apply para buscar dentro del frozenset
        recs = reglas[reglas['antecedents'].apply(lambda x: producto in x)]
        
        if recs.empty:
            print("   ⚠️ No hay suficientes datos históricos para recomendar algo con este producto.")
            return

        # Top 3 recomendaciones
        top_recs = recs.sort_values(by='lift', ascending=False).head(3)
        
        print("🌟 RECOMENDACIONES SUGERIDAS:")
        for idx, row in top_recs.iterrows():
            item = list(row['consequents'])[0]
            confianza = round(row['confidence'] * 100, 1)
            lift = round(row['lift'], 2)
            print(f"   -> {item} (Confianza: {confianza}% | Lift: {lift})")

    # --- PRUEBA AUTOMÁTICA ---
    # Probamos con un producto muy popular: "PLASTERS IN TIN CIRCUS PARADE" (Curitas de Circo)
    # Si no existe, probamos con el primero que encontremos en las reglas.
    
    producto_test = 'PLASTERS IN TIN CIRCUS PARADE'
    
    if producto_test in basket_sets.columns:
        recomendar(producto_test)
    elif not reglas.empty:
        # Fallback al primer producto disponible
        producto_fallback = list(reglas.iloc[0]['antecedents'])[0]
        recomendar(producto_fallback)
            
    print("\n✅ PROCESO FINALIZADO.")

if __name__ == "__main__":
    main()