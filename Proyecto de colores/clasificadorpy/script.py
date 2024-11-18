import pandas as pd

# Cargar el CSV
csv_path = "../dataset/colores_rgb.csv"
data = pd.read_csv(csv_path)

# Guardar como JSON
json_path = "../dataset/colores_rgb.json"
data.to_json(json_path, orient="records", indent=4)

print(f"Archivo JSON generado en {json_path}")