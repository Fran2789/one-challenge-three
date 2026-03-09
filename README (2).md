# 📡 TelecomX – Predicción de Cancelación de Clientes (Churn)

## Propósito del Análisis

Este proyecto tiene como objetivo principal **predecir el churn (cancelación) de clientes** en la empresa TelecomX utilizando técnicas de Machine Learning. A partir de variables demográficas, de servicio y de cuenta, se construyen modelos capaces de identificar qué clientes tienen mayor probabilidad de cancelar, permitiendo a la empresa diseñar estrategias de retención proactivas antes de que ocurra la cancelación.

---

## 📁 Estructura del Proyecto

```
TelecomX/
├── TelecomX_Data.json                        # Datos originales (JSON anidado)
├── TelecomX_Churn_Analysis_Part2.ipynb       # Cuaderno principal (ETL + ML)
├── datos_tratados.csv                        # Datos limpios y transformados
├── README.md                                 # Este archivo
│
└── Gráficos generados al ejecutar el cuaderno:
    ├── churn_distribution.png                # Distribución de la variable objetivo
    ├── boxplots_churn.png                    # Tenure y Gasto Total vs Churn
    ├── cat_churn.png                         # Variables categóricas vs Churn
    ├── correlation_matrix.png                # Matriz de correlación completa
    ├── churn_correlation.png                 # Correlaciones con Churn
    ├── decision_tree.png                     # Visualización del árbol de decisión
    ├── model_comparison.png                  # Comparativa de métricas de modelos
    ├── confusion_matrices.png                # Matrices de confusión por modelo
    ├── lr_coefficients.png                   # Coeficientes de Regresión Logística
    └── rf_feature_importance.png             # Importancia de variables (Random Forest)
```

---

## 🔄 Proceso de Preparación de Datos

### 1. Extracción y Aplanado (ETL)
- El archivo `TelecomX_Data.json` contiene registros anidados con secciones: `customer`, `phone`, `internet`, `account`.
- Cada registro se aplana a un único DataFrame con 21 columnas.

### 2. Limpieza
- La columna `Total` (gasto total) se convierte a numérico; valores no convertibles se marcan como nulos.
- Se eliminan filas con `Churn` vacío (224 registros atípicos).
- La variable `Churn` se convierte a binario: `Yes → 1`, `No → 0`.
- Se eliminan filas con valores nulos restantes.
- **Resultado final:** 7,032 registros limpios.

### 3. Clasificación de Variables

| Tipo | Variables |
|------|-----------|
| **Numéricas** | `tenure`, `Monthly`, `Total`, `SeniorCitizen` |
| **Categóricas** | `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod` |

### 4. Codificación (One-Hot Encoding)
- Se aplica `pd.get_dummies(..., drop_first=True)` a todas las variables categóricas.
- Esto genera 30 features numéricas en total.
- Se usa `drop_first=True` para evitar multicolinealidad (trampa de la variable ficticia).

### 5. Normalización
- Se aplica `StandardScaler` **únicamente** para modelos sensibles a escala:
  - ✅ Regresión Logística
  - ✅ KNN
- Los modelos basados en árboles no requieren normalización:
  - ❌ Árbol de Decisión
  - ❌ Random Forest

### 6. División Train/Test
- División **70% entrenamiento / 30% prueba**.
- Se usa `stratify=y` para mantener la proporción de churn en ambos conjuntos.
- `random_state=5` para reproducibilidad.

---

## 📊 Insights del Análisis Exploratorio

### Desbalance de Clases
- **73.4%** de clientes activos vs **26.6%** que cancelaron.
- El desbalance debe considerarse al interpretar métricas: la exactitud sola puede ser engañosa.

### Variables Clave Identificadas

- **Tipo de Contrato:** Clientes con contrato mes a mes cancelan significativamente más que los de contratos anuales o bianuales.
- **Permanencia (tenure):** Los clientes que cancelan tienen en promedio mucho menos meses de permanencia (diferencia estadísticamente significativa, p < 0.001).
- **Internet Fiber Optic:** Este servicio presenta la mayor tasa de cancelación entre los tipos de internet disponibles.
- **Soporte Técnico:** La ausencia de TechSupport se asocia fuertemente con mayor churn.
- **Método de Pago:** El pago con cheque electrónico tiene mayor tasa de cancelación que los métodos automáticos.

---

## 🤖 Modelos Entrenados y Resultados

| Modelo | Exactitud | Precisión | Recall | F1-Score |
|--------|-----------|-----------|--------|----------|
| **Regresión Logística** | 0.8071 | 0.6667 | 0.5490 | **0.6022** |
| Random Forest | 0.8028 | 0.6651 | 0.5205 | 0.5840 |
| Árbol de Decisión | 0.7877 | 0.6242 | 0.5062 | 0.5591 |
| KNN (k=11) | 0.7749 | 0.5840 | 0.5330 | 0.5573 |
| Dummy (Baseline) | 0.7341 | 0.0000 | 0.0000 | 0.0000 |

**🏆 Modelo ganador: Regresión Logística** (mejor F1-Score = 0.60)

---

## 🚀 Instrucciones de Ejecución

### Requisitos
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Pasos
1. Coloca `TelecomX_Data.json` en el **mismo directorio** que el cuaderno.
2. Abre `TelecomX_Churn_Analysis_Part2.ipynb` en Jupyter Notebook o JupyterLab.
3. Ejecuta las celdas **en orden desde arriba** con `Kernel > Restart & Run All`.
4. El cuaderno generará automáticamente `datos_tratados.csv` y todas las imágenes de gráficos.

### Alternativa: usar datos pre-tratados
Si ya tienes `datos_tratados.csv` de la Parte 1, puedes cargarlo directamente en la sección 2:
```python
df_clean = pd.read_csv('datos_tratados.csv')
```

---

## 🎯 Conclusiones y Recomendaciones Estratégicas

1. **Priorizar retención en clientes nuevos con contrato mensual**: Son el grupo de mayor riesgo.
2. **Mejorar el servicio de fibra óptica**: Alta tasa de cancelación sugiere problemas de calidad o percepción de valor.
3. **Incentivar contratos anuales**: Ofrecer descuentos o beneficios por migrar de mensual a anual.
4. **Activar soporte técnico proactivo**: Los clientes sin TechSupport cancelan más; incluirlo en planes base puede reducir el churn.
5. **Fomentar métodos de pago automático**: Correlacionan con mayor retención.
