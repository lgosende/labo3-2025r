#!/usr/bin/env python3
"""
Combinador de Predicciones - PyTorch LSTM vs AutoGluon
Genera 3 archivos de predicciones combinadas: mínimo, promedio y máximo
"""

import pandas as pd
import numpy as np
from datetime import datetime

def combine_predictions():
    """Combina predicciones de PyTorch y AutoGluon de 3 formas diferentes"""
    
    print("🔄 Combinando predicciones PyTorch LSTM + AutoGluon...")
    print("=" * 60)
    
    # Cargar predicciones
    print("📂 Cargando archivos de predicciones...")
    pytorch_df = pd.read_csv('../output/lstm/predicciones_pytorch_lstm_20002.csv')
    autogluon_df = pd.read_csv('../output/autogluon/predicciones_auto_gl3.csv')
    
    #los valores de pytotch_df los tomo al .90 de su valor.
    #pytorch_df['tn'] = pytorch_df['tn'] * 0.90

    print(f"✅ PyTorch LSTM: {len(pytorch_df)} productos")
    print(f"✅ AutoGluon: {len(autogluon_df)} productos")
    
    # Verificar que tenemos los mismos productos
    pytorch_products = set(pytorch_df['product_id'])
    autogluon_products = set(autogluon_df['product_id'])
    common_products = pytorch_products & autogluon_products
    
    print(f"✅ Productos comunes: {len(common_products)}")
    
    if len(common_products) != 780:
        print("⚠️ Warning: No todos los productos están en ambos archivos")
    
    # Merge de ambos DataFrames
    print("🔗 Combinando datos...")
    merged_df = pd.merge(
        pytorch_df, 
        autogluon_df, 
        on='product_id', 
        suffixes=('_pytorch', '_autogluon')
    )
    
    print(f"📊 Productos combinados: {len(merged_df)}")
    
    # Calcular estadísticas
    print("\n📈 Estadísticas de predicciones:")
    print(f"PyTorch - Min: {merged_df['tn_pytorch'].min():.2f}, Max: {merged_df['tn_pytorch'].max():.2f}, Promedio: {merged_df['tn_pytorch'].mean():.2f}")
    print(f"AutoGluon - Min: {merged_df['tn_autogluon'].min():.2f}, Max: {merged_df['tn_autogluon'].max():.2f}, Promedio: {merged_df['tn_autogluon'].mean():.2f}")
    
    # Calcular totales
    total_pytorch = merged_df['tn_pytorch'].sum()
    total_autogluon = merged_df['tn_autogluon'].sum()
    
    print(f"\n🎯 Totales:")
    print(f"PyTorch LSTM: {total_pytorch:,.2f} toneladas")
    print(f"AutoGluon: {total_autogluon:,.2f} toneladas")
    print(f"Diferencia: {abs(total_pytorch - total_autogluon):,.2f} toneladas ({abs(total_pytorch - total_autogluon)/total_pytorch*100:.1f}%)")
    
    # 1. MÍNIMO: Tomar el valor menor entre las dos predicciones
    print("\n🔽 Generando predicciones MÍNIMAS...")
    min_predictions = merged_df[['product_id']].copy()
    min_predictions['tn'] = np.minimum(merged_df['tn_pytorch'], merged_df['tn_autogluon'])
    min_predictions = min_predictions.sort_values('tn', ascending=False)
    
    # 2. PROMEDIO: Promedio de ambas predicciones
    print("📊 Generando predicciones PROMEDIO...")
    avg_predictions = merged_df[['product_id']].copy()
    avg_predictions['tn'] = (merged_df['tn_pytorch'] + merged_df['tn_autogluon']) / 2
    avg_predictions = avg_predictions.sort_values('tn', ascending=False)
    
    # 3. MÁXIMO: Tomar el valor mayor entre las dos predicciones
    print("🔼 Generando predicciones MÁXIMAS...")
    max_predictions = merged_df[['product_id']].copy()
    max_predictions['tn'] = np.maximum(merged_df['tn_pytorch'], merged_df['tn_autogluon'])
    max_predictions = max_predictions.sort_values('tn', ascending=False)
    
    # Generar timestamp para archivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar archivos
    print("\n💾 Guardando archivos combinados...")
    
    min_filename = f"../output/combined/predicciones_combined_min_{timestamp}.csv"
    avg_filename = f"../output/combined/predicciones_combined_avg_{timestamp}.csv"
    max_filename = f"../output/combined/predicciones_combined_max_{timestamp}.csv"

    min_predictions.to_csv(min_filename, index=False)
    avg_predictions.to_csv(avg_filename, index=False)
    max_predictions.to_csv(max_filename, index=False)
    
    print(f"✅ {min_filename} - Total: {min_predictions['tn'].sum():,.2f} tn")
    print(f"✅ {avg_filename} - Total: {avg_predictions['tn'].sum():,.2f} tn")
    print(f"✅ {max_filename} - Total: {max_predictions['tn'].sum():,.2f} tn")
    
    # Análisis de diferencias
    print("\n📋 Análisis de estrategias:")
    print(f"Estrategia MÍNIMA: {min_predictions['tn'].sum():,.2f} tn (conservadora)")
    print(f"Estrategia PROMEDIO: {avg_predictions['tn'].sum():,.2f} tn (balanceada)")
    print(f"Estrategia MÁXIMA: {max_predictions['tn'].sum():,.2f} tn (optimista)")
    
    # Top 10 productos por estrategia
    print("\n🏆 Top 10 productos - Estrategia PROMEDIO:")
    print(avg_predictions.head(10).to_string(index=False, float_format='%.2f'))
    
    # Comparar diferencias por producto
    print("\n🔍 Productos con mayor diferencia entre modelos:")
    merged_df['diferencia_abs'] = abs(merged_df['tn_pytorch'] - merged_df['tn_autogluon'])
    merged_df['diferencia_pct'] = (merged_df['diferencia_abs'] / merged_df[['tn_pytorch', 'tn_autogluon']].mean(axis=1)) * 100
    
    top_diff = merged_df.nlargest(5, 'diferencia_abs')[['product_id', 'tn_pytorch', 'tn_autogluon', 'diferencia_abs', 'diferencia_pct']]
    print(top_diff.to_string(index=False, float_format='%.2f'))
    
    print(f"\n🎉 Proceso completado! 3 archivos generados con {len(min_predictions)} productos cada uno.")
    
    return min_filename, avg_filename, max_filename

if __name__ == "__main__":
    try:
        min_file, avg_file, max_file = combine_predictions()
        print(f"\n📁 Archivos generados:")
        print(f"   📉 Mínimo: {min_file}")
        print(f"   📊 Promedio: {avg_file}")
        print(f"   📈 Máximo: {max_file}")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()
