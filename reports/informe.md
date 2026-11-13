# Informe de evaluación

Textos de entrenamiento: 16. Textos de prueba: 4. Con tan pocos textos, cada error mueve mucho las métricas; los resultados son orientativos.

## Conjunto de prueba (clase positiva: IA)

| Modelo | Precisión | Recall | F1 | Tasa de falsos positivos |
| --- | --- | --- | --- | --- |
| Línea base (perplejidad) | 0.67 | 1.00 | 0.80 | 0.50 |
| Clasificador | 0.50 | 0.50 | 0.50 | 0.50 |

## Matriz de confusión del clasificador en el conjunto de prueba (umbral 0,5)

| | Predicho humano | Predicho IA |
| --- | --- | --- |
| Real humano | 1 | 1 |
| Real IA | 1 | 1 |

Textos de prueba que quedarían como «indeterminado» con los umbrales calibrados: 1.

## Validación cruzada por grupos (entrenamiento)

| Precisión | Recall | F1 | Tasa de falsos positivos |
| --- | --- | --- | --- |
| 0.78 | 0.88 | 0.82 | 0.25 |

## Coeficientes de la regresión logística (variables estandarizadas)

- perplejidad_media: 0.101
- perplejidad_desv: -0.146
- palabras_distintas: 0.943
- long_oracion_media: -0.156
- long_oracion_desv: -1.259
- repeticion_trigramas: -0.565
