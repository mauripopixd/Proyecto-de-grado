# Prototipo de apoyo para la detección de texto generado por IA en trabajos académicos en español

Prototipo experimental (proyecto de grado, Ingeniería de Sistemas, UNAD). Estima si un texto académico en español
tiene características asociadas a su generación con IA. **El resultado es orientativo: no prueba autoría ni sustenta una sanción.**

## Instalación
```
python -m venv .venv
.venv\Scripts\activate        # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```
La primera vez que se usa la perplejidad se descarga el modelo `PlanTL-GOB-ES/gpt2-base-bne` (necesita internet).

## Pruebas automatizadas
```
pytest
```
Hay 29 pruebas (extractor, modelado, servicio y API). Las de la API requieren `fastapi` y `httpx`, que van en `requirements.txt`.

## Probar el servicio ya (modelo de demostración)
```
python -m scripts.crear_modelo_demo --sin-lm --salida models/modelo.joblib
set USE_LM=0                  # Linux/Mac: export USE_LM=0
uvicorn app.main:app
```
Abra http://127.0.0.1:8000/docs, use `POST /clasificar` con un texto de 100 a 1.000 palabras en español.
**Ese modelo se entrena con datos sintéticos: solo muestra que el servicio funciona de extremo a extremo y la respuesta lo advierte.**
No es un resultado de investigación.

## Flujo con el corpus real
1. Crear `data/corpus/metadatos.csv` con las columnas `id,clase,autor,grupo,archivo` (`clase` es `humano` o `ia`;
   `grupo` agrupa los textos que no deben separarse entre entrenamiento y prueba, por ejemplo todos los de un mismo autor)
   y guardar cada texto (anonimizado, UTF-8) en `data/corpus/textos/`. El corpus no se sube a GitHub.
2. Extraer características: `python -m scripts.extraer_caracteristicas` (agregue `--sin-lm` para usar solo las cuatro de estilo).
3. Entrenar y evaluar: `python -m scripts.entrenar`. Escribe `models/modelo.joblib`, `reports/informe.md` y `reports/metricas.json`.
4. Servir: `uvicorn app.main:app` (sin `USE_LM=0` si el modelo se entrenó con perplejidad).

## Variables de entorno
- `MODEL_PATH`: ruta del modelo (por defecto `models/modelo.joblib`).
- `USE_LM`: con `0` no se carga el modelo de lenguaje (solo sirve con modelos entrenados sin perplejidad).
- `LM_NAME`: modelo de lenguaje alternativo de Hugging Face.

## Estructura
`app/` API, extractor, perplejidad, modelado y servicio · `scripts/` extracción, entrenamiento y modelo de demostración ·
`tests/` pruebas · `reports/` informe de métricas · `models/` modelo entrenado.

## Limitaciones
Corpus pequeño y por conveniencia; solo texto humano frente a texto de IA sin editar; resultados dependen de los asistentes
de IA usados; el prototipo no fue validado con textos mezclados ni de otros idiomas. Se ejecuta en local, sin despliegue.
