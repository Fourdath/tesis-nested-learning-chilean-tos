# Tesis Nested Learning - Chilean Abusive Terms of Service

Este repositorio contiene la implementación experimental asociada a la tesis **Evaluación de Nested Learning para aprendizaje continuo en clasificación incremental de cláusulas potencialmente abusivas en términos de servicio chilenos**.

El objetivo principal del proyecto es construir un flujo experimental reproducible para evaluar una propuesta basada en **Nested Learning** en un escenario de **aprendizaje continuo**, específicamente bajo una configuración de **class-incremental learning** aplicada a cláusulas contractuales en español.

---

## 1. Contexto general del proyecto

La investigación se basa en el problema del **olvido catastrófico** en modelos de aprendizaje profundo. En escenarios de aprendizaje continuo, un modelo debe incorporar nuevas clases o tareas sin degradar significativamente el conocimiento adquirido previamente.

En esta tesis, el problema se aborda usando un corpus de cláusulas potencialmente abusivas en términos de servicio chilenos. La clasificación se organiza en tres grupos principales de clases:

- **Illegal clauses**
- **Dark clauses**
- **Gray clauses**

Estas categorías serán utilizadas como tareas incrementales para evaluar la capacidad del modelo de aprender nueva información sin olvidar las tareas anteriores.

---

## 2. Objetivo del repositorio

Este repositorio busca centralizar:

- La preparación del dataset.
- La creación de tareas incrementales.
- La implementación inicial de Nested Learning.
- La posterior implementación de baselines experimentales.
- La evaluación de métricas de rendimiento y olvido.
- La documentación del protocolo experimental.

La idea es mantener una estructura ordenada y reutilizable, de modo que todas las implementaciones trabajen sobre los mismos datos procesados y los mismos splits experimentales.

---

## 3. Fuente de datos

Los datos originales provienen del repositorio externo:

Nested_Learning_Chilean_Abusive_Terms_of_Service

El dataset original se mantiene localmente en:

`	ext
data/raw/
