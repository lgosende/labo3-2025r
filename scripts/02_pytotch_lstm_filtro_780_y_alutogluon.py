import pandas as pd
import duckdb
# Cargar las predicciones desde el archivo CSV
predicciones = pd.read_csv("predicciones_pytorch.csv")

# Filtrar las predicciones para que solo queden las de los 780 productos
"""
Carga la lista de productos a predecir (780 productos)
"""
print("Cargando productos a predecir...")
con = duckdb.connect(database='../input/db/labo3.duckdb')

query = "SELECT * FROM tb_productos_a_predecir"
productos = con.execute(query).df()
con.close()

productos_ids = productos['product_id'].tolist()
predicciones_filtradas = predicciones[predicciones['product_id'].isin(productos_ids)]

# en las predicciones filtradas "predicciones_filtradas", hay algun registros faltante respecto a al df "productos_ids
# en este caso tomar los ids para los que no tengo predicciones del dataset en "scripts/draft/predicciones_auto_gl3.csv"
# Cargar las predicciones automáticas desde el archivo CSV
predicciones_auto = pd.read_csv("../output/autogluon/draft/predicciones_auto_gl3.csv")

# Identificar los productos faltantes
productos_faltantes = set(productos_ids) - set(predicciones_filtradas['product_id'])

# Filtrar las predicciones automáticas para los productos faltantes
predicciones_faltantes = predicciones_auto[predicciones_auto['product_id'].isin(productos_faltantes)]

# Agregar las predicciones faltantes a las predicciones filtradas
predicciones_filtradas = pd.concat([predicciones_filtradas, predicciones_faltantes], ignore_index=True)


# Guardar el archivo de salida con el campo 'tn' y los 780 productos
predicciones_filtradas.to_csv("../output/lstm/predicciones_pytrch_filtradas.csv", index=False)