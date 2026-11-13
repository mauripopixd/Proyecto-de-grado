# Prototipo de apoyo para la detección de texto generado por IA en trabajos académicos en español

Prototipo experimental del proyecto de grado de Ingeniería de Sistemas (UNAD, curso 202016907, Fase 4).
Estima si un texto académico en español tiene características asociadas a su generación con IA.
**El resultado es orientativo: no prueba autoría ni sustenta una sanción.**

## Estado del prototipo

| Aspecto | Detalle |
| --- | --- |
| Nivel de madurez declarado | TRL 4: componentes integrados y validados en laboratorio (equipo local). |
| Qué incluye | API REST (`/salud`, `/clasificar`), extractor de 6 características, perplejidad con un GPT-2 en español, clasificador de regresión logística con línea base, umbrales calibrados («baja probabilidad», «indeterminado», «alta probabilidad»), 29 pruebas automáticas e informe de evaluación. |
| Qué no incluye | Interfaz web, despliegue en servidor, validación con usuarios reales ni con textos mezclados (humano + IA). |
| Corpus | No se publica (ver [Corpus](#corpus)). Sí se publican la ficha, los scripts que lo construyen y el modelo entrenado. |

## Resultados de la prueba de concepto

Corpus pareado de 20 textos (10 humanos y 10 de IA sobre los mismos 10 temas). Partición por grupo: 16 textos de entrenamiento y 4 de prueba.

| Evaluación | Precisión | Recall | F1 | Falsos positivos |
| --- | --- | --- | --- | --- |
| Validación cruzada por grupos (entrenamiento) | 0,78 | 0,88 | 0,82 | 0,25 |
| Prueba, clasificador (4 textos) | 0,50 | 0,50 | 0,50 | 0,50 |
| Prueba, línea base de perplejidad (4 textos) | 0,67 | 1,00 | 0,80 | 0,50 |

Con 4 textos de prueba, un solo acierto cambia las métricas de forma notable: **son resultados orientativos de una prueba de concepto, no una medida del rendimiento esperado.**
El detalle está en [`reports/informe.md`](reports/informe.md) y [`reports/metricas.json`](reports/metricas.json).
Tiempo de respuesta medido en un portátil con GPU: unos 0,6 s para textos de 400 a 500 palabras (la primera petición tarda más porque carga el modelo de lenguaje).

## Instalación

Requiere Python 3.10 o superior (probado con 3.13 en Windows 11) y conexión a internet la primera vez.

```
python -m venv .venv
.venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

La primera vez que se calcula la perplejidad se descarga el modelo de lenguaje `DeepESP/gpt2-spanish` (unos 260 MB).

## Uso rápido con el modelo entrenado

El repositorio incluye `models/modelo.joblib`, entrenado con el corpus descrito arriba.

```
uvicorn app.main:app
```

Abra <http://127.0.0.1:8000/docs> y use `POST /clasificar` con un texto en español de 100 a 1.000 palabras. También se puede usar el script de prueba, que envía archivos `.txt` y mide el tiempo:

```
python probar_api.py ruta\al\texto.txt
```

Ejemplo de respuesta:

```json
{
  "puntaje_ia": 0.1269,
  "etiqueta": "baja_probabilidad_ia",
  "num_palabras": 410,
  "caracteristicas": { "perplejidad_media": 78.45, "perplejidad_desv": 32.04, "palabras_distintas": 0.4683,
                       "long_oracion_media": 34.17, "long_oracion_desv": 21.50, "repeticion_trigramas": 0.0564 },
  "tiempo_ms": 621.7,
  "modelo_demo": false,
  "advertencia": "Resultado orientativo; no prueba autoría."
}
```

Errores controlados: texto con menos de 100 o más de 1.000 palabras (422), texto que no parece español (422) y solicitud sin el campo `texto` (422). Si no hay modelo entrenado en `MODEL_PATH`, la API responde 503.

## Pruebas automatizadas

```
python -m pytest
```

Son 29 pruebas (extractor, modelado, servicio y API) y se ejecutan sin descargar el modelo de lenguaje.

## Modelo de demostración (opcional)

Para probar el servicio sin el modelo real ni el modelo de lenguaje:

```
python -m scripts.crear_modelo_demo --sin-lm --salida models/modelo_demo.joblib
set MODEL_PATH=models\modelo_demo.joblib   # Linux/Mac: export MODEL_PATH=models/modelo_demo.joblib
set USE_LM=0                               # Linux/Mac: export USE_LM=0
uvicorn app.main:app
```

Ese modelo se entrena con datos sintéticos: solo muestra que el servicio funciona de extremo a extremo, y la respuesta lo advierte (`"modelo_demo": true`).

## Reproducir el entrenamiento con un corpus propio

1. Colocar los textos de origen en `data/fuentes/humanos/` (`H01_T01.pdf` o `.txt`: id, tema) y `data/fuentes/ia/` (`I01_T01_chatgpt.txt` o `.pdf`: id, tema, asistente). Cada tema tiene un texto humano y uno de IA generado con la misma instrucción.
2. `python -m scripts.preparar_corpus` extrae el fragmento de cada texto, quita citas y direcciones, y escribe `data/corpus/textos/`, `data/corpus/metadatos.csv` y `data/ficha_corpus.csv`. Los textos resultantes deben revisarse a mano.
3. `python -m scripts.extraer_caracteristicas` calcula las características (agregue `--sin-lm` para omitir la perplejidad).
4. `python -m scripts.entrenar` entrena, evalúa y escribe `models/modelo.joblib`, `reports/informe.md` y `reports/metricas.json`.

El modelo entrenado depende del modelo de lenguaje usado para la perplejidad; use el mismo valor de `LM_NAME` al entrenar y al servir.

## Variables de entorno

- `MODEL_PATH`: ruta del modelo (por defecto `models/modelo.joblib`).
- `USE_LM`: con `0` no se carga el modelo de lenguaje (solo sirve con modelos entrenados sin perplejidad).
- `LM_NAME`: modelo de lenguaje de Hugging Face (por defecto `DeepESP/gpt2-spanish`).

## Corpus

Los textos humanos son fragmentos de 250 a 500 palabras de trabajos de grado de pregrado de la ECBTI publicados en el repositorio institucional de la UNAD (licencia Creative Commons Reconocimiento-No comercial-Sin obras derivadas); se usan solo internamente para el análisis y no se redistribuyen.
`data/ficha_corpus.csv` registra el enlace, el año y la sección usada de cada texto, sin nombres de autores. Los textos de IA se generaron con una misma instrucción en dos asistentes y no se editaron.

## Estructura

`app/` API, extractor, perplejidad, modelado y servicio · `scripts/` preparación del corpus, extracción, entrenamiento y modelo de demostración ·
`tests/` pruebas · `reports/` informe de métricas · `models/` modelo entrenado · `data/` ficha del corpus · `probar_api.py` prueba de la API.

## Limitaciones

- Corpus muy pequeño (20 textos) y por conveniencia; 4 textos de prueba hacen que las métricas sean anecdóticas.
- Solo se compara texto humano con texto de IA sin editar; no se evaluaron textos mezclados, parafraseados ni de otros idiomas.
- El fragmento humano no siempre es estrictamente la descripción del problema: en algunos trabajos se amplió con secciones contiguas, y un trabajo (H03) no tiene esa sección.
- Los textos humanos y los de IA vienen de fuentes distintas (PDF de repositorio frente a chat), por lo que el modelo podría estar aprendiendo diferencias de origen además del estilo.
- Los resultados dependen de los asistentes y de la instrucción usada para generar los textos de IA.
- El modelo de lenguaje de referencia inicial (`PlanTL-GOB-ES/gpt2-base-bne`) dejó de estar disponible en Hugging Face durante el proyecto y se reemplazó por `DeepESP/gpt2-spanish`.
- Se ejecuta en local, sin despliegue; no debe usarse como evidencia única en procesos académicos o disciplinarios.

## Licencia

MIT (ver `LICENSE`).
