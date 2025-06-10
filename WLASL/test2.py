import pandas as pd

df = pd.read_csv("how2sign_train.csv", delimiter="\t")

# Buscar frases que contengan "hello", "my name is" o "Javi"
matches = df[df["SENTENCE"].str.contains("hello|my name is|javi", case=False, na=False)]

# Ver las frases encontradas
print(matches[["SENTENCE", "VIDEO_ID", "START", "END"]])
