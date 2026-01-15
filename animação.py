import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# ==========================================
# CONFIGURAÇÕES (DEVEM SER IGUAIS À SIMULAÇÃO)
# ==========================================
DIRETORIO_DADOS = 'dados_simulacao' # Pasta onde estão os parquets
RANKING = 1  # 1 = O campeão, 2 = O segundo lugar, etc.

# Física
TIME_STEPS = 1000  # Aumentei para vermos mais tempo de animação (o original era 500)
DT = 0.01
SPACE_SIZE = 1.0
G = 1.0

# ==========================================
# 1. ENCONTRAR O CAMPEÃO
# ==========================================
print(f"Lendo dados para encontrar o Top {RANKING}...")
# Lê apenas as colunas necessárias para economizar RAM
df = pd.read_parquet(DIRETORIO_DADOS, engine='pyarrow', columns=['x', 'y', 'soma_total'])

# Pega os N maiores
top_n = df.nlargest(RANKING, 'soma_total')
campeao = top_n.iloc[RANKING-1] # Pega o último da lista (o ranking escolhido)

start_x = campeao['x']
start_y = campeao['y']
score = campeao['soma_total']

print(f"--- SIMULAÇÃO ESCOLHIDA ---")
print(f"Posição Inicial (Corpo 3): X={start_x:.5f}, Y={start_y:.5f}")
print(f"Pontuação Total: {score:.2f}")
print("Recalculando trajetória completa...")

# ==========================================
# 2. RESSIMULAR A TRAJETÓRIA (Física)
# ==========================================
def calcular_trajetoria(x_ini, y_ini, steps, dt, G):
    # Condições Iniciais (Cópia exata da lógica do Numba)
    pos = np.array([
        [0.25, 0.25],       # Corpo 1
        [0.75, 0.75],       # Corpo 2
        [x_ini, y_ini]      # Corpo 3 (O Campeão)
    ])
    
    vel = np.array([
        [0.05, 0.0],        # Vel 1
        [0.0, -0.05],       # Vel 2
        [0.0, 0.05]         # Vel 3
    ])
    
    masses = np.array([1.0, 1.0, 1.0])
    
    # Histórico para animação: [passo, corpo, coord(x,y)]
    history = np.zeros((steps, 3, 2))
    ejetado = np.array([False, False, False])
    
    for t in range(steps):
        history[t] = pos.copy()
        
        forces = np.zeros_like(pos)
        
        # Cálculo de Forças
        for a in range(3):
            if ejetado[a]: continue
            for b in range(3):
                if a == b or ejetado[b]: continue
                
                r_vec = pos[b] - pos[a]
                dist_sq = np.sum(r_vec**2)
                dist = np.sqrt(dist_sq) + 1e-5
                
                factor = (G * masses[a] * masses[b]) / (dist**3)
                forces[a] += factor * r_vec

        # Integração e Ejeção
        for a in range(3):
            if ejetado[a]: continue
            
            vel[a] += (forces[a] / masses[a]) * dt
            pos[a] += vel[a] * dt
            
            # Se sair do quadrado, marcamos como ejetado (para de interagir)
            # Mas continuamos gravando a posição para ver ele saindo
            if not (0 <= pos[a,0] <= SPACE_SIZE and 0 <= pos[a,1] <= SPACE_SIZE):
                ejetado[a] = True
                
    return history

historico = calcular_trajetoria(start_x, start_y, TIME_STEPS, DT, G)

# ==========================================
# 3. ANIMAÇÃO
# ==========================================
print("Gerando animação...")

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(f"Top {RANKING} - Score: {score:.0f}\nVermelho é o Corpo 3")
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Desenhar o Grid de fundo (Onde os pontos valem mais)
GRID_SIZE = 100
grid_vals = np.arange(1, GRID_SIZE**2 + 1).reshape((GRID_SIZE, GRID_SIZE))
ax.imshow(grid_vals, origin='lower', extent=[0,1,0,1], cmap='Greys', alpha=0.3)

# Elementos gráficos dos 3 corpos
# Corpo 1 (Azul), Corpo 2 (Azul), Corpo 3 (Vermelho - O Protagonista)
colors = ['blue', 'blue', 'red']
sizes = [100, 100, 150]
dots = []
trails = []

for i in range(3):
    # Ponto atual
    dot, = ax.plot([], [], 'o', color=colors[i], ms=8 if i<2 else 12)
    dots.append(dot)
    # Rastro (Trail)
    trail, = ax.plot([], [], '-', color=colors[i], alpha=0.5, lw=1)
    trails.append(trail)

def update(frame):
    # Atualiza cada corpo
    for i in range(3):
        # Posição atual
        x, y = historico[frame, i]
        dots[i].set_data([x], [y])
        
        # Rastro (últimos 50 passos)
        start_trail = max(0, frame - 50)
        trail_x = historico[start_trail:frame+1, i, 0]
        trail_y = historico[start_trail:frame+1, i, 1]
        trails[i].set_data(trail_x, trail_y)
        
    return dots + trails

ani = animation.FuncAnimation(fig, update, frames=TIME_STEPS, interval=20, blit=True)

# Mostra na tela
plt.show()

# Se quiser salvar em vídeo MP4 (precisa do ffmpeg instalado):
# ani.save("trajetoria_campea.mp4", writer='ffmpeg', fps=30)