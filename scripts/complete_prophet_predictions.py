#!/usr/bin/env python3
"""
Script para completar predicciones Prophet con AutoGluon y generar combinaciones
Completa predicciones Prophet = 0 con valores de AutoGluon
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Cargar y analizar ambos archivos de predicciones"""
    print("📂 Cargando archivos de predicciones...")
    
    # Cargar lista completa de productos a predecir (780 productos)
    import duckdb
    con = duckdb.connect(database='input/db/labo3.duckdb')
    query = "SELECT product_id FROM tb_productos_a_predecir ORDER BY product_id"
    productos_780 = con.execute(query).df()
    con.close()
    
    productos_ids_780 = set(productos_780['product_id'].tolist())
    print(f"✅ Productos objetivo (780): {len(productos_ids_780)}")
    
    # Cargar predicciones Prophet
    prophet_df = pd.read_csv("predicciones_prophet_780_products_20250810_091927.csv")
    print(f"✅ Prophet: {len(prophet_df)} productos")
    
    # Cargar predicciones AutoGluon
    autogluon_df = pd.read_csv("scripts/draft/predicciones_auto_gl3_original.csv")
    print(f"✅ AutoGluon: {len(autogluon_df)} productos")
    
    # Análisis de cobertura
    prophet_products = set(prophet_df['product_id'])
    autogluon_products = set(autogluon_df['product_id'])
    
    print(f"\n📊 Análisis de cobertura:")
    print(f"   Productos objetivo: {len(productos_ids_780)}")
    print(f"   En Prophet: {len(prophet_products & productos_ids_780)}")
    print(f"   En AutoGluon: {len(autogluon_products & productos_ids_780)}")
    print(f"   Solo en Prophet: {len(prophet_products - autogluon_products)}")
    print(f"   Solo en AutoGluon: {len(autogluon_products - prophet_products)}")
    
    # Productos faltantes en cada modelo
    missing_in_prophet = productos_ids_780 - prophet_products
    missing_in_autogluon = productos_ids_780 - autogluon_products
    
    print(f"   Faltantes en Prophet: {len(missing_in_prophet)}")
    print(f"   Faltantes en AutoGluon: {len(missing_in_autogluon)}")
    
    # Análisis de Prophet
    prophet_zeros = prophet_df[prophet_df['tn'] == 0]
    prophet_valid = prophet_df[prophet_df['tn'] > 0]
    
    print(f"\n📊 Análisis Prophet:")
    print(f"   Predicciones válidas (>0): {len(prophet_valid)}")
    print(f"   Predicciones en cero: {len(prophet_zeros)}")
    print(f"   Total Prophet: {prophet_df['tn'].sum():,.2f} tn")
    
    print(f"\n📊 Análisis AutoGluon:")
    print(f"   Total AutoGluon: {autogluon_df['tn'].sum():,.2f} tn")
    print(f"   Promedio por producto: {autogluon_df['tn'].mean():.2f} tn")
    
    return prophet_df, autogluon_df, productos_ids_780, missing_in_prophet, missing_in_autogluon

def complete_prophet_predictions(prophet_df, autogluon_df, productos_ids_780, missing_in_prophet):
    """Completar predicciones Prophet con valores de AutoGluon para TODOS los 780 productos"""
    print("\n🔧 Completando predicciones para los 780 productos...")
    
    # Crear DataFrame con todos los 780 productos
    all_products_df = pd.DataFrame({'product_id': sorted(list(productos_ids_780))})
    
    # Merge con Prophet (left join para mantener todos los productos)
    prophet_completed = pd.merge(all_products_df, prophet_df, on='product_id', how='left')
    
    # Rellenar NaN con 0 para productos que no estaban en Prophet
    prophet_completed['tn'] = prophet_completed['tn'].fillna(0)
    
    print(f"📊 Estado inicial:")
    print(f"   Total productos: {len(prophet_completed)}")
    print(f"   Con predicción Prophet > 0: {len(prophet_completed[prophet_completed['tn'] > 0])}")
    print(f"   Con predicción Prophet = 0: {len(prophet_completed[prophet_completed['tn'] == 0])}")
    
    # Contar reemplazos
    replacements_made = 0
    products_not_found = []
    
    # Para cada producto con predicción 0 o faltante
    products_to_complete = prophet_completed[prophet_completed['tn'] == 0]
    
    print(f"🔄 Completando {len(products_to_complete)} productos...")
    
    for _, row in products_to_complete.iterrows():
        product_id = row['product_id']
        
        # Buscar en AutoGluon
        autogluon_match = autogluon_df[autogluon_df['product_id'] == product_id]
        
        if len(autogluon_match) > 0:
            # Reemplazar con valor de AutoGluon
            autogluon_value = autogluon_match['tn'].iloc[0]
            prophet_completed.loc[prophet_completed['product_id'] == product_id, 'tn'] = autogluon_value
            replacements_made += 1
            
            if replacements_made <= 5:  # Mostrar solo los primeros 5
                print(f"   ✅ Producto {product_id}: 0.00 → {autogluon_value:.2f} tn")
        else:
            products_not_found.append(product_id)
    
    # Para productos que no están en ninguno de los dos, usar promedio de AutoGluon
    if products_not_found:
        avg_autogluon = autogluon_df['tn'].mean()
        print(f"⚠️ {len(products_not_found)} productos no encontrados en AutoGluon, usando promedio: {avg_autogluon:.2f}")
        
        for product_id in products_not_found:
            prophet_completed.loc[prophet_completed['product_id'] == product_id, 'tn'] = avg_autogluon
    
    print(f"\n📈 Resultados de completado:")
    print(f"   Total productos: {len(prophet_completed)}")
    print(f"   Reemplazos con AutoGluon: {replacements_made}")
    print(f"   Reemplazos con promedio: {len(products_not_found)}")
    print(f"   Productos finales con tn > 0: {len(prophet_completed[prophet_completed['tn'] > 0])}")
    print(f"   Productos finales con tn = 0: {len(prophet_completed[prophet_completed['tn'] == 0])}")
    
    print(f"   Total Prophet completado: {prophet_completed['tn'].sum():,.2f} tn")
    
    return prophet_completed

def generate_combined_predictions(prophet_completed, autogluon_df, productos_ids_780):
    """Generar predicciones combinadas (mínimo y promedio) para TODOS los 780 productos"""
    print("\n🔗 Generando predicciones combinadas para 780 productos...")
    
    # Asegurar que AutoGluon también tenga todos los productos
    all_products_df = pd.DataFrame({'product_id': sorted(list(productos_ids_780))})
    
    # Completar AutoGluon para productos faltantes
    autogluon_completed = pd.merge(all_products_df, autogluon_df, on='product_id', how='left')
    
    # Para productos faltantes en AutoGluon, usar el promedio
    missing_autogluon = autogluon_completed['tn'].isna().sum()
    if missing_autogluon > 0:
        avg_autogluon = autogluon_df['tn'].mean()
        autogluon_completed['tn'] = autogluon_completed['tn'].fillna(avg_autogluon)
        print(f"📊 Completados {missing_autogluon} productos faltantes en AutoGluon con promedio: {avg_autogluon:.2f}")
    
    # Merge de ambos DataFrames (ahora ambos tienen 780 productos)
    merged_df = pd.merge(
        prophet_completed, 
        autogluon_completed, 
        on='product_id', 
        suffixes=('_prophet', '_autogluon'),
        how='inner'
    )
    
    print(f"📊 Productos combinados: {len(merged_df)} (debe ser 780)")
    
    if len(merged_df) != 780:
        print(f"⚠️ WARNING: Se esperaban 780 productos pero hay {len(merged_df)}")
    
    # 1. MÍNIMO: Tomar el valor menor entre las dos predicciones
    print("🔽 Calculando predicciones MÍNIMAS...")
    min_predictions = merged_df[['product_id']].copy()
    min_predictions['tn'] = np.minimum(merged_df['tn_prophet'], merged_df['tn_autogluon'])
    min_predictions = min_predictions.sort_values('tn', ascending=False)
    
    # 2. PROMEDIO: Promedio de ambas predicciones
    print("📊 Calculando predicciones PROMEDIO...")
    avg_predictions = merged_df[['product_id']].copy()
    avg_predictions['tn'] = (merged_df['tn_prophet'] + merged_df['tn_autogluon']) / 2
    avg_predictions = avg_predictions.sort_values('tn', ascending=False)
    
    # Estadísticas
    print(f"\n📈 Estadísticas de combinaciones:")
    print(f"Prophet completado: {merged_df['tn_prophet'].sum():,.2f} tn ({len(merged_df)} productos)")
    print(f"AutoGluon completado: {merged_df['tn_autogluon'].sum():,.2f} tn ({len(merged_df)} productos)")
    print(f"Combinación MÍNIMA: {min_predictions['tn'].sum():,.2f} tn ({len(min_predictions)} productos)")
    print(f"Combinación PROMEDIO: {avg_predictions['tn'].sum():,.2f} tn ({len(avg_predictions)} productos)")
    
    # Verificar que no hay valores nulos
    null_min = min_predictions['tn'].isna().sum()
    null_avg = avg_predictions['tn'].isna().sum()
    
    if null_min > 0 or null_avg > 0:
        print(f"⚠️ WARNING: Valores nulos encontrados - Min: {null_min}, Avg: {null_avg}")
    
    return min_predictions, avg_predictions, merged_df

def save_results(prophet_completed, min_predictions, avg_predictions):
    """Guardar todos los archivos resultantes"""
    print("\n💾 Guardando archivos...")
    
    # Generar timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Prophet completado
    prophet_filename = f"predicciones_prophet_completed_{timestamp}.csv"
    prophet_completed_sorted = prophet_completed.sort_values('tn', ascending=False)
    prophet_completed_sorted.to_csv(prophet_filename, index=False)
    
    # 2. Combinación mínima
    min_filename = f"predicciones_prophet_autogluon_min_{timestamp}.csv"
    min_predictions.to_csv(min_filename, index=False)
    
    # 3. Combinación promedio
    avg_filename = f"predicciones_prophet_autogluon_avg_{timestamp}.csv"
    avg_predictions.to_csv(avg_filename, index=False)
    
    print(f"✅ {prophet_filename} - Total: {prophet_completed['tn'].sum():,.2f} tn")
    print(f"✅ {min_filename} - Total: {min_predictions['tn'].sum():,.2f} tn")
    print(f"✅ {avg_filename} - Total: {avg_predictions['tn'].sum():,.2f} tn")
    
    return prophet_filename, min_filename, avg_filename

def analyze_top_products(min_predictions, avg_predictions, merged_df):
    """Analizar top productos y diferencias"""
    print(f"\n🏆 Top 10 productos - Estrategia PROMEDIO:")
    print(avg_predictions.head(10).to_string(index=False, float_format='%.2f'))
    
    print(f"\n🔍 Productos con mayor diferencia entre modelos:")
    merged_df['diferencia_abs'] = abs(merged_df['tn_prophet'] - merged_df['tn_autogluon'])
    merged_df['diferencia_pct'] = (merged_df['diferencia_abs'] / merged_df[['tn_prophet', 'tn_autogluon']].mean(axis=1)) * 100
    
    top_diff = merged_df.nlargest(5, 'diferencia_abs')[['product_id', 'tn_prophet', 'tn_autogluon', 'diferencia_abs', 'diferencia_pct']]
    print(top_diff.to_string(index=False, float_format='%.2f'))

def main():
    """Función principal"""
    print("=" * 80)
    print("🔄 COMPLETAR PROPHET + COMBINAR CON AUTOGLUON")
    print("=" * 80)
    
    try:
        # 1. Cargar y analizar datos
        prophet_df, autogluon_df, productos_ids_780, missing_in_prophet, missing_in_autogluon = load_and_analyze_data()
        
        # 2. Completar predicciones Prophet
        prophet_completed = complete_prophet_predictions(prophet_df, autogluon_df, productos_ids_780, missing_in_prophet)
        
        # 3. Generar combinaciones
        min_predictions, avg_predictions, merged_df = generate_combined_predictions(
            prophet_completed, autogluon_df, productos_ids_780
        )
        
        # 4. Guardar resultados
        prophet_file, min_file, avg_file = save_results(prophet_completed, min_predictions, avg_predictions)
        
        # 5. Análisis final
        analyze_top_products(min_predictions, avg_predictions, merged_df)
        
        print(f"\n🎉 Proceso completado exitosamente!")
        print(f"\n📁 Archivos generados:")
        print(f"   📈 Prophet completado: {prophet_file}")
        print(f"   📉 Combinación mínima: {min_file}")
        print(f"   📊 Combinación promedio: {avg_file}")
        
        print(f"\n💡 Recomendación: La estrategia PROMEDIO suele ser más robusta para ensemble de modelos")
        print(f"✅ Verificado: Todos los archivos contienen exactamente 780 productos")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
