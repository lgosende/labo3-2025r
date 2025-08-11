#!/usr/bin/env python3
"""
Prophet-Based Alternative to AutoGluon
Misma lógica pero con Prophet
"""

import pandas as pd
import duckdb
import os
from datetime import datetime
import warnings
from prophet import Prophet
import numpy as np
warnings.filterwarnings('ignore')

def load_products_data():
    """Cargar datos de todos los productos a predecir"""
    print("🔗 Conectando a DuckDB...")
    con = duckdb.connect(database='../input/db/labo3.duckdb')
    
    # Cargar productos a predecir
    print("📋 Cargando productos a predecir...")
    with open("input/tb_productos_a_predecir.txt", "r") as f:
        product_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
    print(f"✅ Productos a predecir: {len(product_ids)}")
    
    # Query optimizada - cargar todos los productos de una vez
    print("📊 Cargando datos de ventas...")
    
    product_ids_str = ','.join(map(str, product_ids))
    
    query = f"""
    SELECT 
        product_id,
        periodo,
        SUM(tn) as tn_total
    FROM ventas_features_final
    WHERE periodo <= 201912 
    AND product_id IN ({product_ids_str})
    GROUP BY product_id, periodo
    ORDER BY product_id, periodo
    """
    
    df_ventas = con.execute(query).df()
    print(f"✅ Datos agregados cargados: {df_ventas.shape}")
    
    con.close()
    return df_ventas, product_ids

def prepare_prophet_data(df_ventas):
    """Preparar datos en formato Prophet"""
    print("\n🧹 Preparando datos para Prophet...")
    
    # Convertir periodo a datetime
    df_ventas['ds'] = pd.to_datetime(df_ventas['periodo'], format='%Y%m')
    df_ventas = df_ventas.rename(columns={'tn_total': 'y'})
    
    print(f"📅 Rango temporal: {df_ventas['ds'].min()} a {df_ventas['ds'].max()}")
    print(f"📦 Productos únicos: {df_ventas['product_id'].nunique()}")
    
    return df_ventas

def train_prophet_model_for_product(product_data):
    """Entrenar modelo Prophet para un producto específico"""
    product_id = product_data['product_id'].iloc[0]
    
    try:
        # Preparar datos para Prophet
        prophet_data = product_data[['ds', 'y']].copy()
        
        # Verificar datos suficientes
        if len(prophet_data) < 12:  # Mínimo 12 meses
            return None
        
        # Configurar Prophet con parámetros conservadores
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05,
            interval_width=0.80
        )
        
        # Entrenar modelo
        model.fit(prophet_data)
        
        # Crear fecha futura para febrero 2020
        future_dates = pd.DataFrame({
            'ds': [pd.to_datetime('2020-02-01')]
        })
        
        # Hacer predicción
        forecast = model.predict(future_dates)
        
        # Extraer predicción
        prediction = max(0, forecast['yhat'].iloc[0])  # No negativos
        
        return {
            'product_id': product_id,
            'tn': prediction,
            'trend': forecast['trend'].iloc[0],
            'seasonal': forecast['yearly'].iloc[0]
        }
        
    except Exception as e:
        print(f"⚠️ Error en producto {product_id}: {e}")
        return None

def train_all_products(df_ventas):
    """Entrenar modelos para todos los productos"""
    print("\n🚀 Entrenando modelos Prophet para todos los productos...")
    
    products = df_ventas['product_id'].unique()
    results = []
    
    total_products = len(products)
    
    for i, product_id in enumerate(products):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"📈 Progreso: {i+1}/{total_products} ({(i+1)/total_products*100:.1f}%)")
        
        # Filtrar datos del producto
        product_data = df_ventas[df_ventas['product_id'] == product_id].copy()
        
        # Entrenar modelo
        result = train_prophet_model_for_product(product_data)
        
        if result is not None:
            results.append(result)
    
    print(f"✅ Modelos entrenados exitosamente: {len(results)}/{total_products}")
    
    return pd.DataFrame(results)

def save_predictions(df_predictions):
    """Guardar predicciones"""
    print("\n💾 Guardando predicciones...")
    
    # Ordenar por predicción descendente
    df_predictions = df_predictions.sort_values('tn', ascending=False)
    
    # Crear archivo de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"../output/prophet/predicciones_prophet_780_products_{timestamp}.csv"
    
    # Guardar solo product_id y tn (igual que AutoGluon)
    output_df = df_predictions[['product_id', 'tn']].copy()
    output_df.to_csv(filename, index=False)
    
    print(f"✅ Archivo guardado: {filename}")
    
    # Estadísticas
    print(f"\n📊 Resumen de predicciones:")
    print(f"   Productos: {len(output_df)}")
    print(f"   Total: {output_df['tn'].sum():,.2f} toneladas")
    print(f"   Promedio por producto: {output_df['tn'].mean():.2f} tn")
    print(f"   Mínimo: {output_df['tn'].min():.2f} tn")
    print(f"   Máximo: {output_df['tn'].max():.2f} tn")
    
    print(f"\n🏆 Top 10 productos:")
    print(output_df.head(10).to_string(index=False, float_format='%.2f'))
    
    return filename

def main():
    """Función principal"""
    print("=" * 70)
    print("📈 PROPHET ALTERNATIVE - PREDICCIÓN 780 PRODUCTOS")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # 1. Cargar datos
        df_ventas, product_ids = load_products_data()
        
        # 2. Preparar datos
        df_prepared = prepare_prophet_data(df_ventas)
        
        # 3. Entrenar modelos
        df_predictions = train_all_products(df_prepared)
        
        # 4. Guardar resultados
        filename = save_predictions(df_predictions)
        
        total_time = datetime.now() - start_time
        
        print(f"\n🎉 Proceso completado en: {total_time}")

        
    except Exception as e:
        print(f"\n💥 Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
