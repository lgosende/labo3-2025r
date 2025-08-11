import pandas as pd

# levanto los 3 mejores modelos.
#"../output\combined\predicciones_combined_min_20250810_141823.csv"
#"../output\2019_avg.csv"
#"../output\autogluon\predicciones_auto_gl3.csv"

# Cargar los 2 CSV
df1 = pd.read_csv("../output/combined/predicciones_combined_min_20250810_084354.csv")

#genero otro df pero con el 97% de las predicciones de df1
df_final_97_df1 = df1.copy()
df_final_97_df1["tn"] = df_final_97_df1["tn"] * 0.97
print(df_final_97_df1.head())
df_final_97_df1.to_csv("../output/combined/03_predicciones_combined_final_97_df1.csv", index=False)





df2 = pd.read_csv("../output/2019_avg.csv")

#df3 = pd.read_csv("../output/autogluon/predicciones_auto_gl3.csv")

# hago un join por product_id entre los 3 dataset por product_id, de manera que me queden por cada product_id 3 columnas tn
merged_df = df1.merge(df2, on="product_id")
#print(merged_df.head())
# promedio tn_x y tn_y
merged_df["tn"] = merged_df[["tn_x", "tn_y"]].mean(axis=1)
# Me quedo con los campos product_id y tn
final_df = merged_df[["product_id", "tn"]]
#print(final_df.head())
#Guardo en un csv
final_df.to_csv("../output/combined/03_predicciones_combined_final.csv", index=False)

#genero otro df pero con el 97% del valor de tn
df_final_97 = final_df.copy()
df_final_97["tn"] = df_final_97["tn"] * 0.97
#print(df_final_97.head())
df_final_97.to_csv("../output/combined/03_predicciones_combined_final_97.csv", index=False)

#genero otro df pero con el 96% del valor de tn
df_final_96 = final_df.copy()
df_final_96["tn"] = df_final_96["tn"] * 0.96
#print(df_final_96.head())
df_final_96.to_csv("../output/combined/03_predicciones_combined_final_96.csv", index=False)

#genero otro df pero con el 95% del valor de tn
df_final_95 = final_df.copy()
df_final_95["tn"] = df_final_95["tn"] * 0.95
#print(df_final_95.head())
df_final_95.to_csv("../output/combined/03_predicciones_combined_final_95.csv", index=False)

