import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

# Configuración
np.random.seed(42)
fs = 1000
duration = 10  # 10 segundos por señal

# =============================================================================
# DEFINICIÓN DE CONDICIONES CARDÍACAS
# =============================================================================
"""
- NORMAL: 60-100 BPM
- BRADICARDIA: < 60 BPM (ritmo lento)
- TAQUICARDIA: > 100 BPM (ritmo rápido)
- ARRITMIA: Variabilidad irregular en el ritmo
"""

# =============================================================================
# GENERACIÓN DE DATASET (30 SEÑALES ECG)
# =============================================================================

def generar_ecg_con_ruido(bpm, duracion, fs, tipo_ruido='medio'):
    """Genera señal ECG con ruido realista"""
    # Generar ECG base con método más estable
    ecg = nk.ecg_simulate(duration=duracion, sampling_rate=fs, heart_rate=bpm, method="ecgsyn")
    
    t = np.arange(0, duracion, 1/fs)
    
    # Agregar diferentes niveles de ruido
    if tipo_ruido == 'bajo':
        ruido_50hz = 0.03 * np.sin(2 * np.pi * 50 * t)  # Ruido de línea
        ruido_muscular = 0.02 * np.random.randn(len(t))
    elif tipo_ruido == 'medio':
        ruido_50hz = 0.05 * np.sin(2 * np.pi * 50 * t)
        ruido_60hz = 0.04 * np.sin(2 * np.pi * 60 * t)
        ruido_muscular = 0.03 * np.random.randn(len(t))
    else:  # alto
        ruido_50hz = 0.08 * np.sin(2 * np.pi * 50 * t)
        ruido_60hz = 0.06 * np.sin(2 * np.pi * 60 * t)
        ruido_muscular = 0.05 * np.random.randn(len(t))
        ruido_baseline = 0.03 * np.sin(2 * np.pi * 0.5 * t)
        return ecg + ruido_50hz + ruido_60hz + ruido_muscular + ruido_baseline
    
    return ecg + ruido_50hz + (ruido_60hz if tipo_ruido == 'medio' else 0) + ruido_muscular

def generar_arritmia(duracion, fs):
    """Genera ECG con arritmia (variabilidad en BPM)"""
    ecg_total = np.array([])
    
    # Simular variaciones irregulares de ritmo
    segmentos_bpm = [75, 110, 65, 95, 120, 70]  # BPM variables
    duracion_segmento = int(duracion / len(segmentos_bpm))  # Convertir a entero
    
    for bpm in segmentos_bpm:
        segmento = nk.ecg_simulate(duration=duracion_segmento, sampling_rate=fs, heart_rate=bpm)
        ecg_total = np.concatenate([ecg_total, segmento])
    
    # Ajustar longitud para que coincida con la duración esperada
    longitud_esperada = int(duracion * fs)
    if len(ecg_total) > longitud_esperada:
        ecg_total = ecg_total[:longitud_esperada]
    elif len(ecg_total) < longitud_esperada:
        padding = np.zeros(longitud_esperada - len(ecg_total))
        ecg_total = np.concatenate([ecg_total, padding])
    
    # Agregar ruido
    ruido = 0.08 * np.random.randn(len(ecg_total))
    return ecg_total + ruido

# Generar dataset
dataset = []
labels = []
bpm_reales = []  # Guardar BPM reales

print("Generando Dataset de 30 Señales ECG...")
print("="*70)

# 10 señales NORMALES (60-100 BPM)
for i in range(10):
    bpm = np.random.randint(65, 95)
    tipo_ruido = np.random.choice(['bajo', 'medio'])
    ecg = generar_ecg_con_ruido(bpm, duration, fs, tipo_ruido)
    dataset.append(ecg)
    labels.append(f'Normal_{bpm}BPM')
    bpm_reales.append(bpm)
    print(f"✓ Señal {i+1}/30: Normal - {bpm} BPM")

# 10 señales BRADICARDIA (< 60 BPM)
for i in range(10):
    bpm = np.random.randint(40, 58)
    tipo_ruido = np.random.choice(['bajo', 'medio'])
    ecg = generar_ecg_con_ruido(bpm, duration, fs, tipo_ruido)
    dataset.append(ecg)
    labels.append(f'Bradicardia_{bpm}BPM')
    bpm_reales.append(bpm)
    print(f"✓ Señal {i+11}/30: Bradicardia - {bpm} BPM")

# 10 señales TAQUICARDIA/ARRITMIA (> 100 BPM o irregular)
for i in range(7):
    bpm = np.random.randint(105, 150)
    tipo_ruido = np.random.choice(['medio', 'alto'])
    ecg = generar_ecg_con_ruido(bpm, duration, fs, tipo_ruido)
    dataset.append(ecg)
    labels.append(f'Taquicardia_{bpm}BPM')
    bpm_reales.append(bpm)
    print(f"✓ Señal {i+21}/30: Taquicardia - {bpm} BPM")

# 3 señales con ARRITMIA real (variabilidad irregular)
for i in range(3):
    ecg = generar_arritmia(duration, fs)
    dataset.append(ecg)
    labels.append(f'Arritmia_Irregular')
    bpm_reales.append(None)  # No tiene BPM fijo
    print(f"✓ Señal {i+28}/30: Arritmia - Ritmo Irregular")

print("="*70)
print("✓ Dataset generado exitosamente!\n")

# Crear categorías
categorias = []
for label in labels:
    if 'Normal' in label:
        categorias.append('Normal')
    elif 'Bradicardia' in label:
        categorias.append('Bradicardia')
    elif 'Taquicardia' in label:
        categorias.append('Taquicardia')
    else:
        categorias.append('Arritmia')

# =============================================================================
# EXTRACCIÓN DE CARACTERÍSTICAS
# =============================================================================

def extract_features(signal, sr=1000):
    """Extrae características completas de señal ECG"""
    signal = np.array(signal, dtype=float)
    
    # Características Temporales
    mean_amp = np.mean(signal) if len(signal) > 0 else 0
    std_amp = np.std(signal) if len(signal) > 0 else 0
    max_amp = np.max(signal) if len(signal) > 0 else 0
    min_amp = np.min(signal) if len(signal) > 0 else 0
    rms = np.sqrt(np.mean(signal**2)) if len(signal) > 0 else 0
    
    # Zero Crossing Rate
    try:
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal))
        if np.isnan(zcr) or np.isinf(zcr):
            zcr = 0
    except:
        zcr = 0
    
    # Características Espectrales
    try:
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
        if np.isnan(spectral_centroid) or np.isinf(spectral_centroid):
            spectral_centroid = 0
    except:
        spectral_centroid = 0
        
    try:
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr))
        if np.isnan(spectral_rolloff) or np.isinf(spectral_rolloff):
            spectral_rolloff = 0
    except:
        spectral_rolloff = 0
        
    try:
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr))
        if np.isnan(spectral_bandwidth) or np.isinf(spectral_bandwidth):
            spectral_bandwidth = 0
    except:
        spectral_bandwidth = 0
    
    # MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=5)
        mfcc_mean = np.mean(mfccs, axis=1)
        # Reemplazar NaN o Inf
        mfcc_mean = np.nan_to_num(mfcc_mean, nan=0.0, posinf=0.0, neginf=0.0)
    except:
        mfcc_mean = np.zeros(5)
    
    # Análisis FFT
    try:
        fft_vals = np.fft.fft(signal)
        fft_freq = np.fft.fftfreq(len(signal), 1/sr)
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        if len(fft_magnitude) > 0:
            dominant_freq = fft_freq_positive[np.argmax(fft_magnitude)]
        else:
            dominant_freq = 0
            
        # Potencia en bandas
        low_band = np.sum(fft_magnitude[(fft_freq_positive >= 0.5) & (fft_freq_positive < 4)])
        mid_band = np.sum(fft_magnitude[(fft_freq_positive >= 4) & (fft_freq_positive < 15)])
        high_band = np.sum(fft_magnitude[(fft_freq_positive >= 15) & (fft_freq_positive < 50)])
        
        # Verificar NaN o Inf
        dominant_freq = 0 if np.isnan(dominant_freq) or np.isinf(dominant_freq) else dominant_freq
        low_band = 0 if np.isnan(low_band) or np.isinf(low_band) else low_band
        mid_band = 0 if np.isnan(mid_band) or np.isinf(mid_band) else mid_band
        high_band = 0 if np.isnan(high_band) or np.isinf(high_band) else high_band
    except:
        dominant_freq = 0
        low_band = 0
        mid_band = 0
        high_band = 0
    
    # Estimación de BPM usando NeuroKit2
    try:
        # Limpiar la señal antes de detectar picos
        cleaned = nk.ecg_clean(signal, sampling_rate=sr)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sr)
        peaks = info['ECG_R_Peaks']
        if len(peaks) > 2:  # Necesitamos al menos 3 picos
            intervals = np.diff(peaks) / sr  # Intervalos en segundos
            # Filtrar intervalos anómalos
            intervals = intervals[(intervals > 0.3) & (intervals < 2.0)]
            if len(intervals) > 0:
                estimated_bpm = 60 / np.mean(intervals)
                bpm_variability = np.std(60 / intervals) if len(intervals) > 1 else 0
            else:
                estimated_bpm = 0
                bpm_variability = 0
        else:
            estimated_bpm = 0
            bpm_variability = 0
    except:
        estimated_bpm = 0
        bpm_variability = 0
    
    # Verificar que no haya NaN o Inf
    estimated_bpm = 0 if np.isnan(estimated_bpm) or np.isinf(estimated_bpm) else estimated_bpm
    bpm_variability = 0 if np.isnan(bpm_variability) or np.isinf(bpm_variability) else bpm_variability
    
    features = {
        'mean': mean_amp,
        'std': std_amp,
        'max': max_amp,
        'min': min_amp,
        'rms': rms,
        'zcr': zcr,
        'spectral_centroid': spectral_centroid,
        'spectral_rolloff': spectral_rolloff,
        'spectral_bandwidth': spectral_bandwidth,
        'dominant_freq': dominant_freq,
        'low_band_power': low_band,
        'mid_band_power': mid_band,
        'high_band_power': high_band,
        'estimated_bpm': estimated_bpm,
        'bpm_variability': bpm_variability,
    }
    
    for i, mfcc_val in enumerate(mfcc_mean):
        features[f'mfcc_{i+1}'] = mfcc_val
    
    return features

print("Extrayendo características del dataset...")
features_list = []
for i, signal in enumerate(dataset):
    features = extract_features(signal, sr=fs)
    features_list.append(features)
    print(f"✓ Procesada señal {i+1}/30", end='\r')

print("\n✓ Extracción completada!\n")

# Crear DataFrame
df_features = pd.DataFrame(features_list)
df_features['label'] = labels
df_features['categoria'] = categorias
df_features['bpm_real'] = bpm_reales  # Agregar BPM real

# Limpiar cualquier NaN o Inf que pueda existir
print("\nLimpiando datos...")
print(f"NaN totales en dataset: {df_features.isna().sum().sum()}")
print(f"Inf totales en dataset: {np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()}")

# Reemplazar NaN e Inf en columnas numéricas
numeric_columns = df_features.select_dtypes(include=[np.number]).columns
df_features[numeric_columns] = df_features[numeric_columns].fillna(0)
df_features[numeric_columns] = df_features[numeric_columns].replace([np.inf, -np.inf], 0)

print(f"NaN después de limpieza: {df_features.isna().sum().sum()}")
print(f"Inf después de limpieza: {np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()}")

# Mostrar resumen
print("="*70)
print("RESUMEN DEL DATASET")
print("="*70)
print(f"\nDimensiones: {df_features.shape}")
print(f"Total de características: {df_features.shape[1] - 3}")  # -3 por label, categoria y bpm_real
print(f"\nDistribución de categorías:")
print(df_features['categoria'].value_counts())
print("\nComparación BPM Real vs Estimado (primeras 15 señales):")
comparacion = df_features[['bpm_real', 'estimated_bpm', 'categoria']].head(15)
comparacion['error'] = abs(comparacion['bpm_real'].fillna(0) - comparacion['estimated_bpm'])
print(comparacion.round(1))

# =============================================================================
# VISUALIZACIÓN DEL DATASET
# =============================================================================

# 1. Visualizar algunas señales de ejemplo
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
fig.suptitle('Ejemplos de Señales ECG del Dataset', fontsize=16, fontweight='bold')

indices_ejemplo = [0, 5, 9, 10, 15, 19, 20, 25, 27, 28, 29, 29]  # Diferentes ejemplos
t = np.arange(0, duration, 1/fs)

for idx, ax in enumerate(axes.flat):
    if idx < len(indices_ejemplo):
        signal_idx = indices_ejemplo[idx]
        ax.plot(t[:3000], dataset[signal_idx][:3000], linewidth=0.8)  # Mostrar 3 segundos
        ax.set_title(f"{labels[signal_idx]}", fontsize=9)
        ax.set_xlabel('Tiempo (s)', fontsize=8)
        ax.set_ylabel('Amplitud', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Color según categoría
        if categorias[signal_idx] == 'Normal':
            ax.set_facecolor('#e8f5e9')
        elif categorias[signal_idx] == 'Bradicardia':
            ax.set_facecolor('#fff3e0')
        elif categorias[signal_idx] == 'Taquicardia':
            ax.set_facecolor('#ffebee')
        else:
            ax.set_facecolor('#f3e5f5')

plt.tight_layout()
plt.show()

# 2. Análisis de BPM estimado
plt.figure(figsize=(14, 6))
colors = {'Normal': 'green', 'Bradicardia': 'blue', 'Taquicardia': 'red', 'Arritmia': 'purple'}
for cat in df_features['categoria'].unique():
    data = df_features[df_features['categoria'] == cat]
    plt.scatter(data.index, data['estimated_bpm'], 
               label=cat, color=colors[cat], alpha=0.6, s=100)

plt.axhline(y=60, color='orange', linestyle='--', label='Límite Bradicardia')
plt.axhline(y=100, color='red', linestyle='--', label='Límite Taquicardia')
plt.xlabel('Índice de Señal', fontsize=12)
plt.ylabel('BPM Estimado', fontsize=12)
plt.title('Distribución de BPM en el Dataset', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Box plot por categoría
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
features_to_plot = ['estimated_bpm', 'bpm_variability', 'spectral_centroid']
titles = ['BPM Estimado', 'Variabilidad BPM', 'Centroide Espectral']

for idx, (feature, title) in enumerate(zip(features_to_plot, titles)):
    df_features.boxplot(column=feature, by='categoria', ax=axes[idx])
    axes[idx].set_title(title, fontsize=12)
    axes[idx].set_xlabel('Categoría', fontsize=10)
    axes[idx].set_ylabel('Valor', fontsize=10)

plt.suptitle('Comparación de Características por Categoría', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. PCA para visualización 2D
print("\nPreparando análisis PCA...")
features_numericas = df_features.drop(['label', 'categoria', 'bpm_real'], axis=1)

# Verificar y mostrar si hay NaN
print(f"NaN encontrados antes de limpieza: {features_numericas.isna().sum().sum()}")

# Reemplazar NaN con 0 (para señales donde no se pudo calcular alguna característica)
features_numericas = features_numericas.fillna(0)

# Verificar después de limpieza
print(f"NaN encontrados después de limpieza: {features_numericas.isna().sum().sum()}")

# Reemplazar cualquier Inf restante
features_numericas = features_numericas.replace([np.inf, -np.inf], 0)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_numericas)

pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_scaled)

plt.figure(figsize=(10, 7))
for cat in df_features['categoria'].unique():
    mask = df_features['categoria'] == cat
    plt.scatter(features_pca[mask, 0], features_pca[mask, 1], 
               label=cat, alpha=0.6, s=100, color=colors[cat])

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varianza)', fontsize=12)
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varianza)', fontsize=12)
plt.title('Visualización PCA del Dataset ECG', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Mapa de calor de correlaciones
plt.figure(figsize=(12, 10))
correlation_matrix = features_numericas.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación de Características', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =============================================================================
# CLASIFICACIÓN CON MACHINE LEARNING
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("\n" + "="*70)
print("ENTRENAMIENTO DE CLASIFICADOR CON MACHINE LEARNING")
print("="*70)

# Preparar datos para entrenamiento
X = df_features.drop(['label', 'categoria', 'bpm_real'], axis=1)
y = df_features['categoria']

print(f"\nDatos de entrenamiento:")
print(f"  Características (X): {X.shape}")
print(f"  Etiquetas (y): {y.shape}")
print(f"  Clases: {y.unique()}")

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDivisión del dataset:")
print(f"  Entrenamiento: {X_train.shape[0]} señales")
print(f"  Prueba: {X_test.shape[0]} señales")

# Entrenar clasificador Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# Evaluar en conjunto de prueba
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*70}")
print(f"RESULTADOS DEL CLASIFICADOR")
print(f"{'='*70}")
print(f"\nPrecisión (Accuracy): {accuracy*100:.2f}%")

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Matriz de confusión:")
cm = confusion_matrix(y_test, y_pred, labels=['Normal', 'Bradicardia', 'Taquicardia', 'Arritmia'])
cm_df = pd.DataFrame(cm, 
                     index=['Normal', 'Bradicardia', 'Taquicardia', 'Arritmia'],
                     columns=['Normal', 'Bradicardia', 'Taquicardia', 'Arritmia'])
print(cm_df)

# Importancia de características
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 características más importantes:")
print(feature_importance.head(10).to_string(index=False))

# Visualizar matriz de confusión
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Matriz de Confusión - Clasificador Random Forest', fontsize=14, fontweight='bold')
plt.ylabel('Clase Real', fontsize=12)
plt.xlabel('Clase Predicha', fontsize=12)
plt.tight_layout()
plt.show()

# Visualizar importancia de características
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importancia', fontsize=12)
plt.title('Top 15 Características Más Importantes', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# SEÑAL INTERACTIVA PARA PRUEBA CON ML
# =============================================================================

print("\n" + "="*70)
print("PRUEBA DE CLASIFICACIÓN CON MACHINE LEARNING")
print("="*70)

# Generar señal de prueba personalizada
print("\nGenerando señal ECG de prueba...")
bpm_prueba = 100  # CAMBIA ESTE VALOR PARA PROBAR (45, 75, 85, 130, etc.)
print(f"BPM configurado: {bpm_prueba}")

ecg_prueba = generar_ecg_con_ruido(bpm_prueba, duration, fs, 'medio')
features_prueba = extract_features(ecg_prueba, sr=fs)

# Convertir a DataFrame para el clasificador
X_prueba = pd.DataFrame([features_prueba])
X_prueba = X_prueba[X.columns]  # Asegurar mismo orden de columnas

# Clasificar usando Machine Learning
clasificacion_ml = clf.predict(X_prueba)[0]
probabilidades = clf.predict_proba(X_prueba)[0]

# Clasificación basada en reglas médicas (para comparar)
def clasificar_por_reglas(bpm, variabilidad=0):
    if variabilidad > 15:
        return "Arritmia"
    elif bpm < 60:
        return "Bradicardia"
    elif bpm > 100:
        return "Taquicardia"
    else:
        return "Normal"

clasificacion_reglas = clasificar_por_reglas(
    features_prueba['estimated_bpm'], 
    features_prueba['bpm_variability']
)

print(f"\n{'='*70}")
print(f"RESULTADOS DE LA SEÑAL DE PRUEBA")
print(f"{'='*70}")
print(f"\n Características de la señal:")
print(f"  BPM Real configurado: {bpm_prueba}")
print(f"  BPM Estimado: {features_prueba['estimated_bpm']:.1f}")
print(f"  Variabilidad BPM: {features_prueba['bpm_variability']:.2f}")
print(f"  RMS: {features_prueba['rms']:.4f}")
print(f"  Centroide Espectral: {features_prueba['spectral_centroid']:.2f} Hz")

print(f"\n🎯 CLASIFICACIONES:")
print(f"  ┌─ Método con REGLAS (if-else): {clasificacion_reglas}")
print(f"  └─ Método con MACHINE LEARNING: {clasificacion_ml}")

print(f"\n📈 Probabilidades del clasificador ML:")
clases = clf.classes_
for clase, prob in zip(clases, probabilidades):
    barra = '█' * int(prob * 50)
    print(f"  {clase:15s} {prob*100:5.1f}% {barra}")

if clasificacion_ml != clasificacion_reglas:
    print(f"\n  NOTA: Los métodos dieron resultados diferentes!")
    print(f"    El ML considera TODAS las características, no solo BPM")
else:
    print(f"\n✓ Ambos métodos coinciden en la clasificación")

# Visualizar señal de prueba
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Señal en tiempo
axes[0].plot(t[:3000], ecg_prueba[:3000], color='darkblue', linewidth=1)
axes[0].set_title(f'Señal ECG de Prueba - {bpm_prueba} BPM → ML: {clasificacion_ml} | Reglas: {clasificacion_reglas}', 
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tiempo (s)')
axes[0].set_ylabel('Amplitud')
axes[0].grid(True, alpha=0.3)

# Espectro de frecuencia
fft_prueba = np.fft.fft(ecg_prueba)
freq_prueba = np.fft.fftfreq(len(fft_prueba), 1/fs)
magnitude_prueba = np.abs(fft_prueba)

axes[1].plot(freq_prueba[:len(freq_prueba)//2], 
            magnitude_prueba[:len(magnitude_prueba)//2], 
            color='purple', linewidth=1)
axes[1].set_xlim(0, 50)
axes[1].set_title('Espectro de Frecuencia', fontsize=12)
axes[1].set_xlabel('Frecuencia (Hz)')
axes[1].set_ylabel('Magnitud')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Guardar dataset (opcional)
# df_features.to_csv('dataset_ecg_30_senales.csv', index=False)
print("\n✓ Análisis completo finalizado!")
print("="*70)
