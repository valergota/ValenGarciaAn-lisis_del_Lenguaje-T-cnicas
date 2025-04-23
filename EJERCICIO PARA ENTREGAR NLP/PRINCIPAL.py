from FUNCIONES import aplicar_stop_word, aplicar_lematizacion, vectorizar, frecuencia_lematizada, Array_StopWords, frecuencia_por_oracion
with open("CorpusLenguajes.txt", "r", encoding="utf-8") as f:
    corpus = f.readlines()
print(corpus)
textos_StopWord = []
textos_Lematizado = []
palabras_Lematizadas = []

print("-"*70)

for texto in corpus:
    t_stopWord = aplicar_stop_word(texto)
    if t_stopWord != "": 
        textos_StopWord.append(t_stopWord)
print(textos_StopWord)

print("-"*70)

for texto in textos_StopWord:
    textos_Lematizado.append(aplicar_lematizacion(texto))
print(textos_Lematizado)

print("-"*70)

print("Matriz TF-IDF:")
df = vectorizar(textos_Lematizado)

print("-"*70)

#preparamos un array de palabras lematizadas para el print
for texto in textos_Lematizado:
      for w in Array_StopWords(texto):
        palabras_Lematizadas.append(w)

frecuencia_lematizada(palabras_Lematizadas)

print("-"*70)

frecuencia_por_oracion(textos_Lematizado)






