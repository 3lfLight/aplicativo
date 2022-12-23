#!/usr/bin/env python
# coding: utf-8
# %%

import pandas as pd
import streamlit as st


# %%
from streamlit_option_menu import option_menu
with st.sidebar:
    selected=option_menu(
    menu_title="CONTENIDO",
    options=["Presentación","Marco Conceptual", "Correlación", "Referencias Bibliográficas"],
    menu_icon="cast")
    
    
if selected=="Presentación":
        st.title(f"Bienvenido a la {selected}")
        st.header('Universidad Nacional de San Agustín de Arequipa') 
        st.header("Escuela Profesional de Ingeniería de Telecomunicaciones")

        st.subheader('Docente : Ingeniero Renzo Bolivar')
        st.subheader("Curso : Computación 1")
        st.subheader("GRUPO C - Nº4")
        st.subheader('Integrantes:') 
        ("Lope Condori Santiago Isaac")
        ("Montalvo Pacori Ivan")
        ("Ramos Catari Joaquin")
        ("Quispe Coila Yampier Edison")
        ("Vilca Medina Milagros Mercedes")
if selected=="Marco Conceptual":
        st.title(f"Bienvenido a la {selected}")
if selected=="Correlación":
        st.title(f"Bienvenido a la {selected}") 
if selected=="Referencias Bibliográficas":
        st.title(f"Bienvenido a la {selected}")        

# %%





# %%



# <center> <h2>Docente : Ingeniero Renzo Bolivar</h2> </center> 

# <center> <h1>Curso : Computación 1</h1> </center> 

# ![linea 1](https://1.bp.blogspot.com/-l1ezJc1oWBU/XRo8vWjDDbI/AAAAAAAAVPc/iy8yM9jcpHY-9gdmEmhhXj9LWwnb2vyEACPcBGAYYCw/s1600/line.png)
# 

# <center> <h2>GRUPO C - Nº4</h2> </center> 
# <h2>INTEGRANTES:  </h2>
# <h2>    
# 
#     Condori Canales Lenin
#     Lope Condori Santiago Isaac
#     Montalvo Pacori Ivan
#     Ramos Catari Joaquin
#     Vilca Medina Milagros Mercedes
# </h2>
# 

# ![linea 1](https://img.wattpad.com/d0459112dc912728bd12cfbbee3e14288328f431/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f776174747061642d6d656469612d736572766963652f53746f7279496d6167652f6675746d7558494e754c713469413d3d2d3638343838313935352e313537626634616632343733343766333839333030353637363834372e706e67)

# <center> <h1>INVESTIGACIÓN FORMATIVA</h1> </center> 
# <center> <h1>PROYECTO FINAL</h1> </center> 
# <center> <h1>PYTHON - Inteligencia Artificial</h1> </center> 

# ![linea 1](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSBBY0Yv9Rgkt2VP8sRGkH3HHe-ACto1_aV452b399eG0isDuDhRx6PlT_u_67LPVt8OQ&usqp=CAU)  
# 

# ## OBJETIVOS

# Los Objetivos de la investigación formativa son:
# 
# - **Competencia Comunicativa** Presentación de sus resultados con lenguaje de programación Python utilizando los archivos Jupyter Notebook.
# - **Competencia Aprendizaje**: con las aptitudes en **Descomposición** (desarticular el problema en pequeñas series de soluciones), **Reconocimiento de Patrones** (encontrar simulitud al momento de resolver problemas), **Abstracción** (omitir información relevante), **Algoritmos** (pasos para resolución de un problema).
# - **Competencia de Trabajo en Equipo**: exige habilidades individuales y grupales orientadas a la cooperación, planificación, coordinación, asignación de tareas, cumplimiento de tareas y solución de conflictos en pro de un trabajo colectivo, utilizando los archivos Jupyter Notebook los cuales se sincronizan en el servidor Gitlab con comandos Git.

# ![linea 1](https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png)

# <center> <h1>Aplicación en IA</h1> </center> 
# <center> <h1>Sistema Recomendador</h1> </center> 

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <div class="alert alert-info">
# El Sistema recomendador deberá encontrar la <strong>compatibilidad o similitud</strong> entre un grupo de personas encuestadas, en las áreas de:
# 
# </div>

# <div class="alert alert-info">
# 
#     
#    -Lugares que desean Conocer
#     
#     
# </div>

# <div class="alert alert-info">
# 
#     
#    La <strong>compatibilidad o similitud</strong> será encontrada con el algoritmo de <strong>Correlación de Pearson</strong> y será verificada con la <strong>La Matrix de Correlación de Pearson con una librería de Python y utilizando una función personal</strong>
#     
# </div>

# 
# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <center> <h1>Base Teórica</h1> </center> 

# ## Análisis de Correlación

# El **análisis de correlación** es el primer paso para construir modelos explicativos y predictivos más complejos.

# <div class="alert alert-info">
# 
#    A menudo nos interesa observar y medir la <strong>relación entre 2 variables numéricas</strong> mediante el análisis de correlación. 
#    Se trata de una de las *técnicas más habituales en análisis de datos* y el primer paso necesario antes de construir cualquier <strong>modelo explicativo o predictivo más complejo</strong>.
#    Para poder tener el  Datset hay que recolectar información a travez de encuentas.
#     
# </div>

# %%


# %%




# ### ¿Cómo se mide la correlación?

# Tenemos el coeficiente de **correlación lineal de Pearson** que se *sirve para cuantificar tendencias lineales*, y el **coeficiente de correlación de Spearman** que se utiliza para *tendencias de aumento o disminución, no necesariamente lineales pero sí monótonas*. 

# ### Correlación de Pearson

# 
# El coeficiente de correlación lineal de Pearson mide una tendencia lineal entre dos variables numéricas.
# 

# 
# Es el método de correlación más utilizado, pero asume que:
# 
#  - La tendencia debe ser de tipo lineal.
#  - No existen valores atípicos (outliers).
#  - Las variables deben ser numéricas.
#  - Tenemos suficientes datos (algunos autores recomiendan tener más de 30 puntos u observaciones).
# 
# Los dos primeros supuestos se pueden evaluar simplemente con un diagrama de dispersión, mientras que para los últimos basta con mirar los datos y evaluar el diseño que tenemos.

# ### Cómo se interpreta la correlación

# El signo nos indica la dirección de la relación, como hemos visto en el diagrama de dispersión.
#  - un valor positivo indica una relación directa o positiva,
#  - un valor negativo indica relación indirecta, inversa o negativa,
#  - un valor nulo indica que no existe una tendencia entre ambas variables (puede ocurrir que no exista relación o que la relación sea más compleja que una tendencia, por ejemplo, una relación en forma de U).

# La magnitud nos indica la fuerza de la relación, y toma valores entre $-1$ a $1$. Cuanto más cercano sea el valor a los extremos del intervalo ($1$ o $-1$) más fuerte será la tendencia de las variables, o será menor la dispersión que existe en los puntos alrededor de dicha tendencia. Cuanto más cerca del cero esté el coeficiente de correlación, más débil será la tendencia, es decir, habrá más dispersión en la nube de puntos.
#  - si la correlación vale $1$ o $-1$ diremos que la correlación es “perfecta”,
#  - si la correlación vale $0$ diremos que las variables no están correlacionadas.

# 
# <center><img src="https://user-images.githubusercontent.com/25250496/204172549-2ccf3be3-a2b3-4b49-9cd4-adb66e28621d.png" width="700" height="4200"></center>
# 
# 
# 

# <center> <h3>Fórmula Coeficiente de Correlación de Pearson</h3> </center>  
# <center> <h3> </h3> </center> 
# $$ r(x,y)=\frac{\sum_{i=1}^{n}(x_{i}-\overline{x})(y_{i}-\overline{y})}{\sqrt{\sum_{i=1}^{n}(x_{i}-\overline{x})^{2}}\sqrt{\sum_{i=1}^{n}(y_{i}-\overline{y})^{2}}}$$

# **Distancia Euclidiana**: La distancia euclidiana es la generalización del __`teorema de Pitágoras`__.

# $$d_{E}(x,y)=\sqrt{\sum_{i=1}^{n}(x_{i}-y_{i})^{2}}$$

# **Regresión Lineal**: La regresión lineal se usa para encontrar una __`relación lineal entre el objetivo y uno o más predictores`__.

# ![que-es-la-regresion-lineal-y-para-que-sirve](https://user-images.githubusercontent.com/25250496/204172072-0fabbfdf-1c4c-4f9b-8f42-505d98b18b71.png)

# ## IMPUTACIÓN DE VALORES NULOS :
# 
# 

# 
# 
#     
#    
#     
# Uno de los problemas más habituales con el que podemos encontrarnos a la hora de trabajar con un conjunto de datos es la existencia de registros con valores nulos. Pudiendo ser necesario imputar un valor a estos registros para poder usarlos en un posterior análisis. Por eso en Scikit-learn existen varias clases con las que se puede realizar la imputación de valores nulos en Python. Clases que utilizan estadísticos básicos como la media, la mediana o la moda y otras que basadas en algoritmos más avanzados como pueden ser los k-vecinos.
# Conjunto de datos con valores nulos
# 
# Antes de realizar la imputación de datos es necesario crear un conjunto de datos al que le falte valores. Algo que se puede hacer con el siguiente ejemplo.
# import numpy as np
# import pandas as pd
# data = pd.DataFrame({'id': [1, 2, 3, 4, 5, 6, 7, 8, 9],
#                      'area': [1, 2, 3, 1, 2, 3, 1, 2, 3],
#                      'age': [32, 30, np.nan, 23, 27, 44, 67, 23, np.nan],
#                      'amount': [102, 121, 343, np.nan, 121, np.nan, 155, 149, 221]})
# 
#    id  area   age  amount
#    
# 0   1     1  32.0   102.0
# 1   2     2  30.0   121.0
# 2   3     3   NaN   343.0
# 3   4     1  23.0     NaN
# 4   5     2  27.0   121.0
# 5   6     3  44.0     NaN
# 6   7     1  67.0   155.0
# 7   8     2  23.0   149.0
# 8   9     3   NaN   221.0
# 
# 
# 
# En este conjunto de datos hay cuatro columnas y en dos de ellas (age y amount) faltan registros. Registros que tienen un valor NaN. Nótese que solamente se han utilizado valores numéricos, esto es porque las clases de imputación no pueden trabajar con cadenas de texto.
# Imputación básica en base a estadísticos
# 
# La clase más sencilla para realizar la imputación de valores nulos es SimpleImputer.Una clase con la que se puede imputar un mismo valor a todos los registros nulos de una columna. Valor que puede ser tanto un estadístico como la media, la mediana o la moda, como un valor constante que se indique. El método de cálculo se indica con la propiedad strategy que puede recibir como valor: mean, median, most_frequent o constant. En el caso de que el valor sea constant es necesario indicar mediante la propiedad fill_value.
# 
# 
# 
# 
# Así se puede imputar por ejemplo los valores nulos en el conjunto de datos anterior utilizado el siguiente código.
# from sklearn.impute import SimpleImputer
# simple = SimpleImputer().fit(data)
# mean = simple.transform(data)
# 
#     id  area        age      amount
# 0  1.0   1.0  32.000000  102.000000
# 1  2.0   2.0  30.000000  121.000000
# 2  3.0   3.0  35.142857  343.000000
# 3  4.0   1.0  23.000000  173.142857
# 4  5.0   2.0  27.000000  121.000000
# 5  6.0   3.0  44.000000  173.142857
# 6  7.0   1.0  67.000000  155.000000
# 7  8.0   2.0  23.000000  149.000000
# 8  9.0   3.0  35.142857  221.000000
# 
# Al no indicar ninguna propiedad, SimpleImputer utiliza la media para imputar los valores. Por lo que el valor empleado en la columna age es 35.14 y en la columna amount es 173.14.
# Seleccionando la moda para imputar valores
# 
# En el caso de que queramos que el valor sea entero, como en el resto de los registros, una alternativa es usar la moda. Lo que se muestra en el siguiente ejemplo asignando a la propiedad strategy el valor most_frequent.
# simple = SimpleImputer(strategy='most_frequent')
# mode = simple.fit_transform(data)
# 
#     id  area   age  amount
# 0  1.0   1.0  32.0   102.0
# 1  2.0   2.0  30.0   121.0
# 2  3.0   3.0  23.0   343.0
# 3  4.0   1.0  23.0   121.0
# 4  5.0   2.0  27.0   121.0
# 5  6.0   3.0  44.0   121.0
# 6  7.0   1.0  67.0   155.0
# 7  8.0   2.0  23.0   149.0
# 8  9.0   3.0  23.0   221.0
# 
# Ahora el valor usado en la columna age es 23, mientras que para amount es 121.
# 
# 
# 
# 
# Uso de k-vecinos para la imputación de valores nulos
# 
# El uso de un valor constante a la hora de realizar la imputación de valores nulos es una buena opción en muchas ocasiones. Pero en otras puede ser un problema ya que puede existir una gran variabilidad en los datos. Por eso se puede usar en su lugar un método como el de k-vecinos con el que se puede identificar los registros más similares al que presenta los valores nulos y asignarle la media de sus vecinos. Lo que posiblemente produzca un valor más parecido al real.
# 
# La clase que implementa este método es KNNImputer. A continuación, podemos ver que pasaría si asignamos a cada registro con un valor nulo el valor de su vecino más cercano. Lo que se puede conseguir indicando mediante la propiedad n_neighbors de la clase que solamente tenga en cuenta un vecino. Lo que se muestra en el siguiente ejemplo.
# from sklearn.impute import KNNImputer
# knn = KNNImputer(n_neighbors=1)
# neighbors = knn.fit_transform(data)
# 
#     id  area   age  amount
# 0  1.0   1.0  32.0   102.0
# 1  2.0   2.0  30.0   121.0
# 2  3.0   3.0  23.0   343.0
# 3  4.0   1.0  23.0   343.0
# 4  5.0   2.0  27.0   121.0
# 5  6.0   3.0  44.0   343.0
# 6  7.0   1.0  67.0   155.0
# 7  8.0   2.0  23.0   149.0
# 8  9.0   3.0  44.0   221.0
# 
# Obviamente se puede aumentar el número de vecinos para obtener una mejor estimación de los valores.
# knn = KNNImputer(n_neighbors=5)
# neighbors = knn.fit_transform(data)
#     
# 
#     id  area   age  amount
# 0  1.0   1.0  32.0   102.0
# 1  2.0   2.0  30.0   121.0
# 2  3.0   3.0  37.4   343.0
# 3  4.0   1.0  23.0   191.0
# 4  5.0   2.0  27.0   121.0
# 5  6.0   3.0  44.0   181.6
# 6  7.0   1.0  67.0   155.0
# 7  8.0   2.0  23.0   149.0
# 8  9.0   3.0  36.8   221.0
#     

# 
# ### Función Reshape

# El método que podemos usar en NumPy para redimensionar los vectores es la función reshape. Una función que es clave conocer para trabajar de forma eficaz con NumPy. Veamos a continuación como se puede usar la función reshape de NumPy a través de diferentes ejemplos.
# La función reshape de NumPy
# 
# En la documentación de NumPy se pude ver que la función reshape tiene la siguiente forma
# 
# np.reshape(a, newshape, order='C')
# 
# donde
# 
#     a: el vector de NumPy que se desea redimensionar.
#     newshape: una tupla, en el caso de que se desee convertir en un vector 23 o 30, o un valor entero, cuando el vector de destino 1D.
#     order: un valor opción en el que se indica el orden de llenado de los vectores. Los valores disponibles para esta propiedad son:
#     * C: lee y escribe los elementos del vector por filas, estilo como en C.
#     * F: lee y escribe los elementos del vector por columnas, estilo como en FORTRAN
#     * A: lee y escribe los elementos del vector según el orden que tengan esos en memoria.
# 
# El nuevo tamaño tiene que ser compatible con el original
# 
# Un punto importante es que el nuevo tamaño que se le indique a la función reshape tiene que ser compatible con el original. Esto es, si en el vector original hay 10 elementos, tiene que haber necesariamente 10 en el de nuevo. Si no es así se producirá un error. Esto es así porque reshape no permite omitir elementos del vector original ni dejar elementos sin valor en el de destino.
# Uso básico de reshape en NumPy
# 
# En el caso de que tengamos un vector en memoria se puede usar reshape para crear un vector 2D con los mismos datos. Para ello solamente se tiene que llamar a la función y pasar el vector original y una tupla con las nuevas dimensiones como parámetro. Así para convertir un vector en un objeto 2D se puede usar
# 
# 
# import numpy as np
# arr = np.arange(12)
# np.reshape(arr, (2, 6))
# 
# array([[ 0,  1,  2,  3,  4,  5],
#        [ 6,  7,  8,  9, 10, 11]])
# 
# Con lo que se ha convertido una vector en una matriz de 6 por dos. Otra alternativa sería crear una matriz de 2 por seis, lo que se puede conseguir mediante
# np.reshape(arr, (6, 2))
# 
# array([[ 0,  1],
#        [ 2,  3],
#        [ 4,  5],
#        [ 6,  7],
#        [ 8,  9],
#        [10, 11]])
# 
# En el caso de que queramos crear un objeto 3D, simplemente es necesario pasar una tupla de tres elementos.
# np.reshape(arr, (2, 2, 3))
# 
# array([[[ 0,  1,  2],
#         [ 3,  4,  5]],
# 
#        [[ 6,  7,  8],
#         [ 9, 10, 11]]])
# 
# Convertir matrices 2D o 3D en vectores
# 
# Para convertir una matriz tanto de 2D como 3D en un vector se ha de indicar un escalar con el número de elementos. Aunque, en estos casos, es más cómodo utilizar -1, ya que convertirá la matriz en un vector sin necesidad de conocer su tamaño. Lo que evita posibles errores. Esto es, una matriz 3D se puede transformar en una vector con las dos líneas que se muestra a continuación.
# arr3d = np.reshape(arr, (2, 2, 3))
# np.reshape(arr3d, 12)
# np.reshape(arr3d, -1)
# 
# array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
# 
# Tipos de llenado de los vectores
# 
# Como ya se ha explicado al principio existen tres tipos de llenado para los vectores. Por filas, al estilo C
# 
# Publicidad
# 
# 
# arr3d = np.arange(9)
# np.reshape(arr3d, (3,3), order='C')
# 
# array([[0, 1, 2],
#        [3, 4, 5],
#        [6, 7, 8]])
# 
# Por columnas, al estilo FORTRAN
# np.reshape(arr3d, (3,3), order='F')
# 
# array([[0, 3, 6],
#        [1, 4, 7],
#        [2, 5, 8]])
# 
# Y en base a la posición en memoria.
# np.reshape(arr3d, (3,3), order='A')
# 
# array([[0, 3, 6],
#        [1, 4, 7],
#        [2, 5, 8]])
# 
# En estos ejemplos no se puede hacer una diferencia entre el tipo C en base a la posición en memoria. Pero sí que es posible observar esto cuando se realiza una transformación, por ejemplo, una transposición de la matriz.
# arr = np.arange(9)
# arr2d = np.reshape(arr, (3,3), order='C').T
# arr2d
# 
# array([[0, 3, 6],
#        [1, 4, 7],
#        [2, 5, 8]])
# 
# En este caso el método C leerá los datos por filas
# np.reshape(arr2d, 9, order='C')
# 
# array([0, 3, 6, 1, 4, 7, 2, 5, 8])
# 
# Mientras que el método A recuperará los datos tal como están ubicados en memoria.
# 
# np.reshape(arr2d, -1, order='A')
# 
# array([0, 1, 2, 3, 4, 5, 6, 7, 8])
# 
# Los resultados de reshape son vistas
# 
# Es importante tener en cuenta que siempre que es posible reshape devuelve una vista del objeto original. No un nuevo objeto con los valores. Por lo que, en caso de modificar los valores en la vista, también se modificarán en el objeto original y viceversa. Efecto que se puede observar en el siguiente ejemplo.
# arr = np.arange(4)
# arr2d = np.reshape(arr, (2,2))
# arr2d[1][1] = 9
# print(arr)
# print(arr2d)
# 
# [0 1 2 9]
# [[0 1]
#  [2 9]]
# 
# 

# ### Numpy.sqrt 

# 
# numpy.sqrt(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'sqrt'>
# 
# Return the non-negative square-root of an array, element-wise.
# 
# Parameters
# 
#     xarray_like
# 
#         The values whose square-roots are required.
#     outndarray, None, or tuple of ndarray and None, optional
# 
#         A location into which the result is stored. If provided, it must have a shape that the inputs broadcast to. If not provided or None, a freshly-allocated array is returned. A tuple (possible only as a keyword argument) must have length equal to the number of outputs.
#     wherearray_like, optional
# 
#         This condition is broadcast over the input. At locations where the condition is True, the out array will be set to the ufunc result. Elsewhere, the out array will retain its original value. Note that if an uninitialized out array is created via the default out=None, locations within it where the condition is False will remain uninitialized.
#     **kwargs
# 
#         For other keyword-only arguments, see the ufunc docs.
# 
# Returns
# 
#     yndarray
# 
#         An array of the same shape as x, containing the positive square-root of each element in x. If any element in x is complex, a complex array is returned (and the square-roots of negative reals are calculated). If all of the elements in x are real, so is y, with negative elements returning nan. If out was provided, y is a reference to it. This is a scalar if x is a scalar.
# 
# 

# 
# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <center> <h1>Propuesta</h1> </center> 

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# ## 1.- Dataset

# <div class="alert alert-info">
# 
#     
#    Para poder tener el  <strong>Datset</strong> hay que recolectar información con una encuenta elaborada por nosotros.
#     
# </div>

# #### Encuesta Lugares a los que nos gustaría viajar:

# La encuesta la realizamos en Google-Form donde se solicitara escoger un Lugar al que le gustaría viajar.
# - Donde si escoge 1 es el que menos le gusta hasta 5 que es el que mas le gusta (escala de liker)

# #### Formulario de Lugares a los que me gustaría viajar (Preguntas)
# ![1.png](attachment:1.png)
# 

# ![5.png](attachment:5.png)

# 
# 
# 
# 
# #### Formulario de Lugares a los que me gustaría viajar 
# 
# ![2.png](attachment:2.png)

# ![3.png](attachment:3.png)

# ![4.png](attachment:4.png)

# #### Formulario de Lugares a los que me gustaría viajar(Preprocesamiento)

# %%


#Importamos librerias para Ciencia de Datos y Machine Learning
import numpy as np
import pandas as pd
import seaborn as sns


# %%


#archivo CSV separado por comas

data = pd.read_csv("Lugares.csv")

#leer  lineas
data


# %%


data.shape


# %%


data.dtypes


# ## IMPUTACIÓN DE VALORES NULOS 

# %%


data1= data.fillna(data.mean())
data1


# 
# 

# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# ## 2.- Correlación de Pearson  (Similitud)

# %%


n = data1[data1.columns[1:]].to_numpy()
m = data1[data1.columns[0]].to_numpy()
print(n)
print(m)


# ## 3.- Correlación en Pandas

# %%


n.T


# %%


df1 = pd.DataFrame(n.T, columns = m)
df1


# %%


m_corr = df1.corr()
m_corr


# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# ## 4.- Matrix de Correlación

# %%


m_corr_pandas = np.round(m_corr, 
                       decimals = 2)  
  
m_corr_pandas


# %%


m_corr_pandas


# %%


pandas= m_corr_pandas.unstack()

print(pandas.sort_values(ascending=False)[range(len(n),((len(n)+4)))])


# ## Gráfica de Calor

# <div class="alert alert-info">
# 
#     
#    **HALLAR**: a partir de la matriz de correlación en  <strong>Pandas</strong> .
#     
#    **Instalar** : `matplotlib` `seaborn`
#     
# </div>

# %%


sns.heatmap(m_corr_pandas)


("5.- RESULTADOS") 
("Los resultados de similitud obtenidos en **Lugares que me gustaría viajar** según la tabla de **Correlación** con los siguientes encuestados:")
("1. cosmic99latte@gmail.com y milam@gmail.com  obtienen el **PRIMER** indice mas alto de similitud con 72% ")
("2. esmelizeth@gmail.com   y  milam@gmail.com obtienen el **SEGUNDO** indice mas alto de similitud con 70%")


("Validación - Matrix de Correlación")

# Se realiza la validación de los resultados obtenidos con la   `Matriz de Correlación de Pearson` en `Numpy` 
#  

# ### VALIDACION:

# %%


n = data1[data1.columns[1:]].to_numpy()
m = data1[data1.columns[0]].to_numpy()
print(n)
print(m)


# %%


import math
corr_grupal=[]

def correlaciongrupal(x,y):
    xprom, yprom=x.mean(), y.mean()
    arriba=np.sum((x-xprom)*(y-yprom))
    abajo=np.sqrt(np.sum((x-xprom)**2)*np.sum((y-yprom)**2))
    return arriba/abajo

for columna in range (len(m)):
        for fila in range(len(m)):
            datos=data1.loc[[columna,fila],:]
            datos2=datos[datos.columns[1:]].to_numpy()
            corr_grupal.append(correlaciongrupal(datos2[0],datos2[1]))
            
corre_grupal=np.array(corr_grupal).reshape(len(m),len(m))
correlacion=pd.DataFrame(corre_grupal,m,m)
correlacion
    


# %%


pandas= correlacion.unstack()

print(pandas.sort_values(ascending=False)[range(len(n),((len(n)+4)))])


# %%


import matplotlib.pyplot as plt
sns.heatmap(correlacion)


# ![linea 2](https://user-images.githubusercontent.com/19308295/115926262-2fb62980-a448-11eb-8189-c2f10e499944.png)

# <center> <h1>Conclusiones</h1> </center> 

# ##  ¿Se validó o no los resultados?
# 
#    -Si se validaron los resultados a través de numpy.

# ## Los resultados Validados son:
#  

# %%


#RESULTADO 1:
pandas= m_corr_pandas.unstack()

print(pandas.sort_values(ascending=False)[range(len(n),((len(n)+4)))])

#RESULTADO 2:
pandas1= correlacion.unstack()

print(pandas1.sort_values(ascending=False)[range(len(n),((len(n)+4)))])

("CONCLUSIONES")
("¿Es efectivo el metodo de correlación de pearson?")
("-Si, es efectivo.")

("Correlacion de Pearson y Regresion lineal, ¿Cuál es su relación?")

("Teniendo en cuenta (x,y), si la correlacion entre las dos variables es fuerte la regresión permite encontrar la función matemática que las relacione adecuadamente.")

# %%
("Referencias")

("Profesor de Matematicas: `John Gabriel Muñoz Cruz`https://www.linkedin.com/in/jgmc")
 
("Interactive Chaos: https://interactivechaos.com/es/manual/tutorial-de-numpy/la-funcion-reshape")

("Analitycs Lane: https://www.analyticslane.com/2021/04/05/numpy-la-funcion-reshape-de-numpy-con-ejemplos/")
 
("NumPy:https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html")
 
("Correlación de Python: http://blog.espol.edu.ec/estg1003/correlacion-con-python-ejercicio-estacion-meteorologica/")
 
("Instituto Programador: https://www.instintoprogramador.com.mx/2020/11/tutorial-de-matriz-de-correlacion-de.html")

("Mapas de calor :https://www.delftstack.com/es/howto/seaborn/correlation-heatplot-seaborn-python/")

("Stack overflow :https://stackoverflow.com/questions/51077418/what-is-the-most-efficient-way-of-doing-square-root-of-sum-of-square-of-two-numb")
 
("Linear Models with R by Julian J.Faraway libro")
 
("OpenIntro Statistics: Fourth Edition by David Diez, Mine Çetinkaya-Rundel, Christopher Barr libro")

("Introduction to Machine Learning with Python: A Guide for Data Scientists libro")
 
("Points of Significance: Association, correlation and causation. Naomi Altman & Martin Krzywinski Nature Methods")

("https://user-images.githubusercontent.com/19308295/115926252-2b8a0c00-a448-11eb-9d9c-b43beaf0ff68.png")


# %%





# %%





# %%




