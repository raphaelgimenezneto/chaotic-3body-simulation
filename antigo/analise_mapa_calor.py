import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# --- CONFIGURAÇÃO IGUAL À SIMULAÇÃO ---
DIRETORIO_DADOS = 'dados_simulacao' 
X_MIN, X_MAX = 0.0, 0.0
Y_MIN, Y_MAX = 0.0, 0.0

SPACE_SIZE = 1.0   # tem que bater com a simulação
USE_ZOOM = False   # True = zoom, False = ver tudo

if USE_ZOOM:
    X_MIN, X_MAX = 0.4, 0.6
    Y_MIN, Y_MAX = 0.4, 0.6
else:
    X_MIN, X_MAX = 0.0, SPACE_SIZE
    Y_MIN, Y_MAX = 0.0, SPACE_SIZE

# Se você manteve o GRID_SIZE da visualização em 200, 
# agora esses 200 pixels representam apenas a área de 0.5 de largura.
# Isso significa que a resolução dobrou (ou quadruplicou em área)!
GRID_SIZE = 100

print(f"Carregando dados...")
df = pd.read_parquet(DIRETORIO_DADOS, engine='pyarrow')

print("Calculando histogramas na área de zoom...")

# DEFININDO O RANGE DO HISTOGRAMA
# Isso garante que só vamos "binarizar" os dados dentro da janela de zoom
range_zoom = [[Y_MIN, Y_MAX], [X_MIN, X_MAX]]

counts, _, _ = np.histogram2d(df['y'], df['x'], bins=GRID_SIZE, range=range_zoom)
soma_vals, _, _ = np.histogram2d(df['y'], df['x'], bins=GRID_SIZE, weights=df['soma_total'], range=range_zoom)
soma_sq, _, _ = np.histogram2d(df['y'], df['x'], bins=GRID_SIZE, weights=df['soma_total']**2, range=range_zoom)

# ... (Cálculo de média e variância IDÊNTICO ao anterior) ...
mask = counts > 0
media = np.zeros((GRID_SIZE, GRID_SIZE))
variancia = np.zeros((GRID_SIZE, GRID_SIZE))
media[mask] = soma_vals[mask] / counts[mask]
variancia[mask] = (soma_sq[mask] / counts[mask]) - (media[mask]**2)
variancia[variancia < 0] = 0

# Classificação
vals_media = media[mask]
vals_var = variancia[mask]
if len(vals_media) > 0:
    limite_media = np.median(vals_media)
    limite_var = np.median(vals_var)
else:
    limite_media = 0; limite_var = 0

class_map = np.full((GRID_SIZE, GRID_SIZE), -1, dtype=int)
is_high_mean = media >= limite_media
is_high_var = variancia >= limite_var

class_map[mask & is_high_mean & ~is_high_var] = 0 
class_map[mask & is_high_mean & is_high_var] = 1
class_map[mask & ~is_high_mean & is_high_var] = 2
class_map[mask & ~is_high_mean & ~is_high_var] = 3

# PLOTAGEM COM EXTENSÃO CORRETA
print("Gerando gráficos...")
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Define os limites dos eixos para os números ficarem certos no gráfico
extent = [X_MIN, X_MAX, Y_MIN, Y_MAX]

im1 = axs[0].imshow(media, origin='lower', cmap='viridis', extent=extent)
axs[0].set_title(f"Média (Zoom: {X_MIN}-{X_MAX})")
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(variancia, origin='lower', cmap='magma', extent=extent)
axs[1].set_title("Variância")
fig.colorbar(im2, ax=axs[1])

class_map_masked = np.ma.masked_where(class_map == -1, class_map)
cmap = ListedColormap(['green', 'gold', 'red', 'blue'])

im3 = axs[2].imshow(class_map_masked, origin='lower', cmap=cmap, extent=extent, interpolation='nearest')
axs[2].set_title("Classificação")
axs[2].set_facecolor('black')

# ... (Legenda igual ao anterior) ...
labels = ['Alta Média / Baixa Var', 'Alta Média / Alta Var', 'Baixa Média / Alta Var', 'Baixa Média / Baixa Var']
patches = [mpatches.Patch(color=['green', 'gold', 'red', 'blue'][i], label=labels[i]) for i in range(4)]
axs[2].legend(handles=patches, loc='upper right', fontsize='8')

plt.tight_layout()
plt.show()