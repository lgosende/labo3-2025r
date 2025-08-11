#!/usr/bin/env python3
"""
LSTM PyTorch Script - Predicción de ventas por cliente-producto
Alternativa a TensorFlow que detecta mejor las GPUs
"""

import duckdb
import pandas as pd
import numpy as np
import warnings
import multiprocessing as mp
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# PyTorch para LSTM (alternativa a TensorFlow)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch disponible")
except ImportError:
    print("❌ PyTorch no está instalado. Instalar con: pip install torch")
    PYTORCH_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configurar GPU con PyTorch (más confiable que TensorFlow)
print("🔧 Configurando GPU con PyTorch...")
print("=" * 50)

if PYTORCH_AVAILABLE:
    # Verificar CUDA disponible
    cuda_available = torch.cuda.is_available()
    print(f"🎮 CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"🔌 GPU devices: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Configurar GPU por defecto
        device = torch.device("cuda:0")
        print(f"✅ Usando GPU: {torch.cuda.get_device_name(0)}")
        
        # Test básico
        try:
            x = torch.randn(100, 100).to(device)
            y = torch.mm(x, x)
            print(f"🧪 Test GPU exitoso: tensor shape {y.shape}")
            USE_GPU = True
        except Exception as e:
            print(f"❌ Test GPU falló: {e}")
            device = torch.device("cpu")
            USE_GPU = False
    else:
        print("⚠️ CUDA no disponible, usando CPU")
        device = torch.device("cpu")
        USE_GPU = False
else:
    device = torch.device("cpu")
    USE_GPU = False

print("=" * 50)
print(f"🎯 Dispositivo final: {device}")
print("=" * 50)

# Configuración
USE_REGRESSORS = True
PRODUCT_ID_TEST = 20002
USE_ALL_PRODUCTS = True

# Configuración dinámica según GPU/CPU
def get_optimized_config():
    if USE_GPU:
        return {
            'batch_size': 128,
            'max_parallel_models': 4,
            #'n_cores': 4,
            'epochs': 50,
            'sequence_length': 6,  # Reducido para 6 meses de datos
            'hidden_size': 64,
            'num_layers': 2,
            'learning_rate': 0.001
        }
    else:
        return {
            'batch_size': 32,
            'max_parallel_models': min(mp.cpu_count() - 1, 20),
            'epochs': 30,
            'sequence_length': 4,  # Reducido para pocos períodos
            'hidden_size': 32,
            'num_layers': 1,
            'learning_rate': 0.001
        }

# Variables numéricas para regresores
NUMERIC_COLS = [
    #'antiguedad_cliente',
    #'antiguedad_producto',
    #'cust_request_qty',
    #'cust_request_tn',
    #'crecimiento_consistente',
    #'dias_desde_primera_venta_cliente',
    #'dias_desde_ultima_venta_cliente',
    #'es_invierno',
    #'es_verano',
    #'is_max_tn_12m',
    #'is_max_tn_3m',
    #'is_max_tn_6m',
    #'is_min_tn_12m',
    #'is_min_tn_3m',
    #'is_min_tn_6m',
    #'ma_request_3m',
    #'ma_request_6m',
    #'ma_tn_12m',
    #'ma_tn_3m',
    #'ma_tn_6m',
    #'mes',
    #'trimestre',
    #'clientes_total_producto',
    #'es_cliente_principal_producto',
    #'es_producto_principal_cliente',
    #'meses_desde_ultima_compra',
    #'patron_ciclico_3m',
    #'patron_ciclico_6m',
    #'productos_total_cliente_periodo',
    #'promedio_anual',
    'promedio_historico_mes',
    #'ratio_vs_promedio_otros_productos_cliente',
    #'tn_total_cliente_periodo',
    #'tn_total_producto_periodo'
]

class TimeSeriesDataset(Dataset):
    """Dataset personalizado para secuencias temporales"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LSTMModel(nn.Module):
    """Modelo LSTM con PyTorch"""
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Tomar la última salida
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        
        return out

def load_data_from_db():
    """Carga datos desde DuckDB"""
    print("🔗 Conectando a la base de datos...")
    con = duckdb.connect(database='input/db/labo3.duckdb')
    
    query = f"""
    SELECT 
        customer_id,
        product_id,
        periodo,
        periodo_fecha,
        tn,
        {', '.join(NUMERIC_COLS)}
    FROM ventas_features_final
    WHERE periodo <= 201912
    ORDER BY customer_id, product_id, periodo
    """
    
    if not USE_ALL_PRODUCTS:
        query = query.replace("WHERE periodo", f"WHERE product_id = {PRODUCT_ID_TEST} AND periodo")
    
    df = con.execute(query).df()
    con.close()
    
    print(f"📊 Datos cargados: {df.shape}")
    return df

def create_sequences_pytorch(data, sequence_length, use_regressors=True):
    """Crea secuencias para PyTorch LSTM"""
    X, y = [], []
    
    if len(data) < sequence_length + 2:
        return np.array(X), np.array(y)
    
    for i in range(sequence_length, len(data) - 1):
        if use_regressors:
            # Combinar tn con features
            tn_seq = data['tn'].iloc[i-sequence_length:i].values
            features_seq = data[NUMERIC_COLS].iloc[i-sequence_length:i].values
            
            # Crear secuencia combinada
            combined_seq = np.column_stack([
                tn_seq.reshape(-1, 1),
                features_seq
            ])
            X.append(combined_seq)
        else:
            # Solo tn
            X.append(data['tn'].iloc[i-sequence_length:i].values.reshape(-1, 1))
        
        # Target (siguiente período)
        y.append(data['tn'].iloc[i + 1])
    
    return np.array(X), np.array(y)

def train_pytorch_lstm_model(args):
    """Entrena modelo LSTM con PyTorch"""
    if not PYTORCH_AVAILABLE:
        return None
        
    customer_id, product_id, subset, config = args
    
    try:
        # ⚡ OPTIMIZACIÓN: subset ya viene filtrado, no necesitamos filtrar de nuevo
        subset = subset.sort_values('periodo_fecha')
        
        if len(subset) < config['sequence_length'] + 2:
            return None
        
        # Normalizar datos
        scaler_tn = MinMaxScaler()
        subset['tn_scaled'] = scaler_tn.fit_transform(subset[['tn']])
        
        if USE_REGRESSORS:
            scaler_features = StandardScaler()
            subset[NUMERIC_COLS] = scaler_features.fit_transform(subset[NUMERIC_COLS].fillna(0))
        
        # Preparar datos para secuencias
        data_for_seq = subset[['tn_scaled'] + NUMERIC_COLS] if USE_REGRESSORS else subset[['tn_scaled']]
        data_for_seq.columns = ['tn'] + NUMERIC_COLS if USE_REGRESSORS else ['tn']
        
        # Crear secuencias
        X, y = create_sequences_pytorch(data_for_seq, config['sequence_length'], USE_REGRESSORS)
        
        if len(X) == 0:
            return None
        
        # Crear dataset y dataloader
        dataset = TimeSeriesDataset(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            num_workers=0  # 0 para evitar problemas en Windows
        )
        
        # Crear modelo
        input_size = X.shape[2]  # Número de features
        model = LSTMModel(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers']
        ).to(device)
        
        # Configurar entrenamiento
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Entrenar modelo
        model.train()
        total_loss = 0
        
        for epoch in range(config['epochs']):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss = epoch_loss / len(dataloader)
            
            # Early stopping simple
            if epoch > 10 and total_loss < 0.001:
                break
        
        # Hacer predicción
        model.eval()
        with torch.no_grad():
            # Usar la última secuencia para predecir
            last_sequence = torch.FloatTensor(X[-1:]).to(device)
            prediction_scaled = model(last_sequence).cpu().numpy()[0][0]
        
        # Desnormalizar
        prediction = scaler_tn.inverse_transform([[prediction_scaled]])[0][0]
        prediction = max(0, prediction)  # No negativos
        
        return {
            'customer_id': customer_id,
            'product_id': product_id,
            'predicted_tn': prediction,
            'final_loss': total_loss
        }
        
    except Exception as e:
        print(f"⚠️ Error PyTorch {customer_id}-{product_id}: {e}")
        return None

def run_pytorch_models_parallel(df, combinations):
    """Ejecuta modelos PyTorch en paralelo"""
    config = get_optimized_config()
    
    print(f"🚀 Iniciando entrenamiento PyTorch de {len(combinations)} modelos...")
    print(f"💻 Dispositivo: {device}")
    print(f"🧵 Modelos paralelos: {config['max_parallel_models']}")
    print(f"📏 Secuencia: {config['sequence_length']}")
    print(f"📦 Batch size: {config['batch_size']}")
    
    # ⚡ OPTIMIZACIÓN: Pre-agrupar datos por combinación (evita filtros repetidos)
    print("⚡ Preparando datos agrupados...")
    df_grouped = df.groupby(['customer_id', 'product_id'])
    
    # Preparar argumentos de forma optimizada
    args_list = []
    for customer_id, product_id in combinations:
        try:
            # Obtener subset del grupo pre-calculado (mucho más rápido)
            subset = df_grouped.get_group((customer_id, product_id)).copy()
            args_list.append((customer_id, product_id, subset, config))
        except KeyError:
            # Si no existe la combinación, saltarla
            continue
    
    print(f"📊 Args preparados: {len(args_list)}")
    
    # Ejecutar en paralelo
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=config['max_parallel_models']) as executor:
        future_to_args = {
            executor.submit(train_pytorch_lstm_model, args): args 
            for args in args_list
        }
        
        for i, future in enumerate(as_completed(future_to_args)):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                
                if (i + 1) % 5 == 0:
                    elapsed = time.time() - start_time
                    print(f"📈 Progreso: {i+1}/{len(combinations)} ({(i+1)/len(combinations)*100:.1f}%) - {elapsed:.1f}s")
                    
            except Exception as e:
                print(f"⚠️ Error: {e}")
    
    total_time = time.time() - start_time
    print(f"✅ Completado en {total_time:.1f}s - Modelos exitosos: {len(results)}")
    
    return results

def prepare_data_and_run():
    """Función principal optimizada"""
    print("📊 Preparando datos...")
    
    # Cargar datos
    df = load_data_from_db()
    df['periodo_fecha'] = pd.to_datetime(df['periodo_fecha'])
    
    # Obtener combinaciones válidas
    combinations = df[['customer_id', 'product_id']].drop_duplicates()
    config = get_optimized_config()
    min_periods = config['sequence_length'] + 2
    
    print(f"🔍 Análisis de datos:")
    print(f"   Períodos mínimos requeridos: {min_periods}")
    print(f"   Secuencia LSTM: {config['sequence_length']}")
    print(f"   Total combinaciones: {len(combinations)}")
    
    # ⚡ OPTIMIZACIÓN: Usar operaciones vectorizadas de pandas (muchísimo más rápido)
    print("⚡ Analizando períodos con operaciones vectorizadas...")
    
    # Contar períodos por combinación de una sola vez
    period_counts_series = df.groupby(['customer_id', 'product_id']).size()
    
    print(f"   Períodos por combinación - Min: {period_counts_series.min()}, Max: {period_counts_series.max()}, Promedio: {period_counts_series.mean():.1f}")
    
    # Filtrar combinaciones válidas directamente
    valid_combinations_mask = period_counts_series >= min_periods
    valid_combinations = period_counts_series[valid_combinations_mask].index.tolist()
    
    print(f"✅ Combinaciones válidas: {len(valid_combinations)}")
    
    if len(valid_combinations) == 0:
        print("❌ No hay combinaciones válidas")
        print(f"💡 Sugerencia: Reducir sequence_length a {period_counts_series.max() - 3} o menos")
        return
    
    # Ejecutar modelos
    results = run_pytorch_models_parallel(df, valid_combinations)
    
    # Agregar predicciones
    if results:
        df_results = pd.DataFrame(results)
        final_predictions = df_results.groupby('product_id')['predicted_tn'].sum().reset_index()
        final_predictions.columns = ['product_id', 'tn']
        final_predictions = final_predictions.sort_values('tn', ascending=False)
        
        print(f"\n🎯 Resultados finales:")
        print(f"Total tn predichas: {final_predictions['tn'].sum():.2f}")
        
        # Guardar
        filename = f"predicciones_pytorch_lstm_{PRODUCT_ID_TEST}.csv"
        final_predictions.to_csv(filename, index=False)
        print(f"💾 Guardado en: {filename}")
        
        print("\n🏆 Top 5 predicciones:")
        print(final_predictions.head().to_string(index=False))

def main():
    """Función principal"""
    if not PYTORCH_AVAILABLE:
        print("❌ PyTorch no está disponible. Instalar con:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    config = get_optimized_config()
    
    print("=" * 60)
    print("🔥 PYTORCH LSTM - PREDICCIÓN DE VENTAS")
    print("=" * 60)
    print(f"🎯 Producto: {PRODUCT_ID_TEST}")
    print(f"💻 Dispositivo: {device}")
    print(f"🔧 GPU habilitada: {USE_GPU}")
    print(f"📊 Configuración: {config}")
    print("=" * 60)
    
    try:
        prepare_data_and_run()
        print("\n🎉 Proceso completado!")
        
    except Exception as e:
        print(f"💥 Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
