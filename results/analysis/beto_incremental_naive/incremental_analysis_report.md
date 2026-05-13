# Análisis del baseline incremental BETO

## Registro del avance realizado

Durante esta jornada se avanzó en la implementación inicial del protocolo experimental para la tesis, enfocado en evaluar aprendizaje continuo en un escenario de clasificación incremental de cláusulas potencialmente abusivas en términos de servicio chilenos. En primer lugar, se reorganizó la base de datos original del corpus en una estructura incremental compuesta por tres tareas: `Illegal`, `Dark` y `Gray`. Cada tarea quedó separada en subconjuntos de entrenamiento, validación y prueba, permitiendo construir un flujo de entrenamiento secuencial y evaluación acumulativa.

Posteriormente, se implementó un cargador del dataset incremental, encargado de leer los archivos procesados y transformar las etiquetas en formato multi-label mediante vectores multi-hot de tamaño 24. Esto fue necesario porque una cláusula puede estar asociada a más de una etiqueta al mismo tiempo. Luego, se incorporó el tokenizador de BETO (`dccuchile/bert-base-spanish-wwm-cased`), modelo basado en BERT y preentrenado para español, con el objetivo de representar adecuadamente el lenguaje del corpus.

También se configuró el entorno de entrenamiento con GPU, logrando que PyTorch reconociera correctamente la tarjeta NVIDIA GeForce RTX 4060 Laptop GPU mediante CUDA. Con esto, se validó la generación de embeddings con BETO, obteniendo representaciones de tamaño 768 para cada cláusula. A partir de estos embeddings, se implementó un clasificador multi-label con una capa lineal de salida de 24 logits, uno por cada etiqueta global del dataset.

Antes del entrenamiento incremental, se realizó una prueba individual sobre la primera tarea (`Illegal`) para verificar que el modelo pudiera entrenar correctamente, calcular pérdida, actualizar pesos y producir métricas de validación y prueba. Una vez validado este flujo, se implementó un baseline secuencial ingenuo con BETO, entrenando el mismo modelo de forma consecutiva en las tareas `Illegal → Dark → Gray`, sin mecanismos adicionales para proteger el conocimiento aprendido previamente. El objetivo de este baseline fue obtener una primera medición del olvido catastrófico bajo el protocolo incremental.

Este reporte resume el comportamiento del baseline secuencial ingenuo. El modelo se entrena en una tarea a la vez y luego se evalúa acumulativamente sobre las tareas vistas hasta ese momento.

## Matriz de micro-F1

| after_training | eval_task_1_illegal | eval_task_2_dark | eval_task_3_gray |
| --- | --- | --- | --- |
| Task 1 (illegal) | 0.6734 |  |  |
| Task 2 (dark) | 0.5367 | 0.7722 |  |
| Task 3 (gray) | 0.3383 | 0.7640 | 0.7085 |

## Matriz de macro-F1

| after_training | eval_task_1_illegal | eval_task_2_dark | eval_task_3_gray |
| --- | --- | --- | --- |
| Task 1 (illegal) | 0.3248 |  |  |
| Task 2 (dark) | 0.4288 | 0.4301 |  |
| Task 3 (gray) | 0.2644 | 0.4563 | 0.5347 |

## Retención y olvido final

| task | initial_micro_f1 | final_micro_f1 | micro_drop | micro_forgetting | initial_macro_f1 | final_macro_f1 | macro_drop | macro_forgetting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Task 1 (illegal) | 0.6734 | 0.3383 | 0.3351 | 0.3351 | 0.3248 | 0.2644 | 0.0604 | 0.1644 |
| Task 2 (dark) | 0.7722 | 0.7640 | 0.0082 | 0.0082 | 0.4301 | 0.4563 | -0.0262 | 0.0000 |
| Task 3 (gray) | 0.7085 | 0.7085 | 0.0000 | 0.0000 | 0.5347 | 0.5347 | 0.0000 | 0.0000 |

## Lectura de resultados

Los resultados muestran una primera evidencia de olvido catastrófico en el baseline secuencial. La caída más relevante ocurre en la Task 1 (`Illegal`), cuyo rendimiento en micro-F1 disminuye desde 0.6734, luego de entrenar inicialmente en esa tarea, hasta 0.3383 después de completar el entrenamiento secuencial en las tres tareas. Esto corresponde a una pérdida absoluta de 0.3351 en micro-F1, lo que representa un nivel alto de olvido.

En macro-F1, la Task 1 también presenta una caída, pasando de 0.3248 a 0.2644 al final del proceso. Sin embargo, se observa que el comportamiento del macro-F1 no siempre coincide directamente con el micro-F1, debido al desbalance de clases y a que el macro-F1 pondera cada clase de manera uniforme. Por esta razón, ambas métricas deben ser reportadas y analizadas de forma complementaria.

Para la Task 2 (`Dark`), el olvido es prácticamente inexistente. El micro-F1 disminuye levemente desde 0.7722 hasta 0.7640, con una pérdida de solo 0.0082. En macro-F1, incluso se observa una mejora desde 0.4301 hasta 0.4563. Esto puede deberse a que la Task 2 fue entrenada justo antes de la Task 3 y, por lo tanto, estuvo menos expuesta al deterioro acumulado que la Task 1.

La Task 3 (`Gray`) no presenta olvido, lo cual es esperable, ya que corresponde a la última tarea entrenada. En este caso, sus métricas iniciales y finales coinciden, porque no existen tareas posteriores que puedan afectar su rendimiento.

## Lectura automática

- Task 1 (illegal): olvido micro-F1 alto (0.3351) y olvido macro-F1 moderado (0.1644).
- Task 2 (dark): olvido micro-F1 sin olvido relevante (0.0082) y olvido macro-F1 sin olvido relevante (0.0000).
- Task 3 (gray): olvido micro-F1 sin olvido relevante (0.0000) y olvido macro-F1 sin olvido relevante (0.0000).

## Interpretación preliminar

La corrida preliminar permitió validar el pipeline completo de entrenamiento incremental y evaluación acumulativa. El resultado más importante es que el baseline secuencial ingenuo muestra una pérdida clara de rendimiento en la primera tarea después de aprender tareas posteriores. Esto constituye una primera medición experimental del olvido catastrófico dentro del escenario incremental propuesto para la tesis.

Sin embargo, estos resultados todavía deben entenderse como preliminares. El entrenamiento se realizó con una configuración inicial de pocas épocas por tarea, por lo que aún no corresponde a una versión final ni optimizada del baseline. Aun así, el experimento cumple un objetivo metodológico importante: demostrar que el protocolo incremental funciona, que las métricas se calculan correctamente y que el escenario construido permite observar degradación del rendimiento en tareas antiguas.

## Siguiente paso

La siguiente etapa será robustecer el entrenamiento del baseline BETO para obtener resultados más sólidos y defendibles. Para ello, se propone implementar una versión más exigente del entrenamiento incremental, incorporando mayor número de épocas, early stopping, uso de mixed precision, gradient accumulation, ajuste de umbral de decisión, manejo del desbalance de clases y una utilización más intensiva de la GPU.

El objetivo de esta mejora es evitar que la aparición de olvido catastrófico pueda atribuirse solamente a un entrenamiento demasiado simple. Una vez obtenido un baseline secuencial más robusto, este servirá como punto de comparación para implementar y evaluar mecanismos orientados a mejorar la retención de conocimiento, incluyendo Nested Learning, EWC y replay.



nota Fecha 09/05/2026