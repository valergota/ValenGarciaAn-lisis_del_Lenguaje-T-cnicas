import nltk
nltk.download('stopwords')
from nltk import pos_tag, word_tokenize, FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer #tuve que instalar la libreria scikit-learn
import pandas as pd
import string

#aplicamos la funcion aplicar_stop_words para eliminar las palabras repettidas en el corpus
def aplicar_stop_word(corpus):
    tokens = word_tokenize(corpus)
    stopwords_en = stopwords.words("english")
    corpus_limpio = [w.lower() for w in tokens if w not in ["corpus", "lematizar","quitarStopwords_eng","word_tokenize"]  #aplicamos la funcion for x(words) in(en) tokens not in(no presentes en) w(words)
                                                                and w.lower() not in stopwords_en #saco las stopwords y paso todas las w(words) a minuscula
                                                                and w not in string.punctuation #saco los simbolos de puntuacion
                                                                and w.isalpha()] #verifico que todos los w(words) sean palabras
    corpus_limpio = ' '.join(corpus_limpio) #convierto la lista de w(words) en un string donde se formaran oraciones separadas por un ' '
    return corpus_limpio

#aplicamos la lematizacion para las palabras en su forma base
def aplicar_lematizacion(corpus):
    tokens = word_tokenize(corpus) #aplico la funcion word_tokenize que separa las palabras de las oraciones en tokens
    lemmatizer = WordNetLemmatizer() #dejo en claro que estoy inicializando la lematizacion
    texto_lema = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens] #lematizo cada palabra de una lista de tokens para obtener el tipo de palabra que es
    return ' '.join(texto_lema) #uno todas las palabras de la lista en un string separado por espacios

#aplicamos la funcion para graficar la frecuencia de palabras
def frecuencia_lematizada (texto_lematizado):
    frecuencia=FreqDist(texto_lematizado) #ojo, freqdist me dice cuantas veces aparece un elemento en una lista
    print(frecuencia.most_common(52)) #me muestra la cantidad de plabras presentes en el corpus, las cuales se que hay 52 porque me dice en la cantidad de columnas de la tabla del TF-IDF
    # Graficar la distribución de frecuencias
    plt.figure(figsize=(10, 6)) #me dice el tamaño del grafico
    frecuencia.plot(20, show=True)  #elijo las 20 palabras mas frecuentes en el corpus
    plt.show()
    return frecuencia

#aplicamos la funcion para separar el string en palabras, es decir, volverlo una lista
def Array_StopWords (texto_lematizado):
    tokens = word_tokenize(texto_lematizado)
    return tokens

#lo hice pero no lo utilice en la pestaña principal porque no supe como aplicarlo
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


#aplicamos la funcion para realizar el tf-idf
def vectorizar(corpus):
    vectorizer = TfidfVectorizer() #transformo el texto en una matriz numerica
    X = vectorizer.fit_transform(corpus) #aca cada oracion esta representada por una fila, y cada palabra corresponde a una columna en especifico
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    print("\nMatriz TF-IDF:")
    print(df)  
    return df #devuelve el dataframe (tener el cuenta que dataframe es como una lista con filas y columnas)
    
#aplicamos la funcion para saber con cuanta frecuencia se repiten las palabras por oracion en el corpus
def frecuencia_por_oracion(corpus_lema):
    for i, oracion in enumerate(corpus_lema, 1): #iteracion por sobre cada una de las oraciones
        tokens = oracion.split() #divido las oraciones en palabras separadas por un espacio
        frecuencia = FreqDist(tokens) #calculola frecuencia y la aparicion de cada palabra
        print(f"\n Oración {i}: {oracion}")
        print("Palabras más frecuentes:")
        for palabra, freq in frecuencia.most_common():
            print(f"  {palabra} = {freq}")



