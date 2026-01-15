import numpy as np
import pandas as pd
import os
from numba import njit, prange

# -----------------------------
# CONFIGURAÇÕES
# -----------------------------
GRID_SIZE = 200
TIME_STEPS = 600
DT = 0.01
SPACE_SIZE = 1.0
G = 1.0
NUM_SIMULATIONS = 500000
BATCH_SIZE = 50000  # Aumentado para otimizar escrita em disco
OUTPUT_DIR = "dados_simulacao_zoom"

# Configuração de ZOOM
USE_ZOOM = False  # Mude para True quando quiser focar numa área

if USE_ZOOM:
    # Exemplo: Zoom no centro
    X_MIN, X_MAX = 0.4, 0.6
    Y_MIN, Y_MAX = 0.4, 0.6
else:
    # Espaço completo
    X_MIN, X_MAX = 0.0, SPACE_SIZE
    Y_MIN, Y_MAX = 0.0, SPACE_SIZE

# -----------------------------
# FUNÇÕES DE SIMULAÇÃO (OTIMIZADAS)
# -----------------------------

@njit(inline='always')
def get_cell_index(x, y, grid_size, space_size):
    # Converte posição contínua para índice discreto
    xi = int(x / space_size * grid_size)
    yi = int(y / space_size * grid_size)
    
    # Clamp (garante que não saia do array)
    if xi >= grid_size: xi = grid_size - 1
    elif xi < 0: xi = 0
    
    if yi >= grid_size: yi = grid_size - 1
    elif yi < 0: yi = 0
        
    return xi, yi

@njit(parallel=True, fastmath=True)
def simular_varias_posicoes(posicoes_iniciais, grid_size, space_size, time_steps, dt, G):
    N = posicoes_iniciais.shape[0]
    resultados = np.zeros(N, dtype=np.float64)
    
    # Massas (Assumindo 1.0 para otimizar multiplicação)
    # Se quiser massas diferentes, crie um array masses = np.array([...])
    
    for i in prange(N):
        # OTIMIZAÇÃO: Alocar zeros e preencher é mais rápido que np.array([a,b,c]) no loop
        pos_x = np.zeros(3); pos_y = np.zeros(3)
        vel_x = np.zeros(3); vel_y = np.zeros(3)
        fx = np.zeros(3); fy = np.zeros(3)
        ejetado = np.zeros(3, dtype=np.bool_) # Array booleano
        
        # Configuração Inicial
        # Corpos 0 e 1 fixos, Corpo 2 varia
        pos_x[0] = 0.25; pos_x[1] = 0.75; pos_x[2] = posicoes_iniciais[i, 0]
        pos_y[0] = 0.25; pos_y[1] = 0.75; pos_y[2] = posicoes_iniciais[i, 1]
        
        vel_x[0] = 0.05; vel_x[1] = 0.0;  vel_x[2] = 0.0
        vel_y[0] = 0.0;  vel_y[1] = -0.05; vel_y[2] = 0.05
        
        total_sum = 0.0
        num_ejetados = 0

        for _ in range(time_steps):
            # Zerar forças
            fx[:] = 0.0
            fy[:] = 0.0
            
            # 1. Calcular Forças Gravitacionais
            for a in range(3):
                if ejetado[a]: continue
                for b in range(3):
                    if a == b or ejetado[b]: continue
                    
                    dx = pos_x[b] - pos_x[a]
                    dy = pos_y[b] - pos_y[a]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq) + 1e-5 # Epsilon para evitar div/0
                    
                    # F = G * m1 * m2 / r^3 * vec(r)
                    # Como massas são 1.0, simplificamos:
                    factor = G / (dist * dist * dist)
                    
                    fx[a] += factor * dx
                    fy[a] += factor * dy

            # 2. Integração (Euler Explícito - Como no original)
            for a in range(3):
                if ejetado[a]: continue
                
                # F = ma => a = F/m (m=1.0)
                vel_x[a] += fx[a] * dt
                vel_y[a] += fy[a] * dt
                
                pos_x[a] += vel_x[a] * dt
                pos_y[a] += vel_y[a] * dt
                
                # Checar limites
                if not (0.0 <= pos_x[a] <= space_size and 0.0 <= pos_y[a] <= space_size):
                    ejetado[a] = True
                    num_ejetados += 1
            
            if num_ejetados == 3: break
            
            # 3. Somar Valor do Grid
            # OTIMIZAÇÃO: Cálculo matemático em vez de lookup em matriz
            for a in range(3):
                if not ejetado[a]:
                    xi, yi = get_cell_index(pos_x[a], pos_y[a], grid_size, space_size)
                    # Equivalente a buscar na matriz arange reshape
                    valor_celula = (yi * grid_size) + xi + 1 
                    total_sum += valor_celula
        
        resultados[i] = total_sum

    return resultados

# -----------------------------
# LOOP PRINCIPAL
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- INICIANDO SIMULAÇÃO ---")
    print(f"Modo Zoom: {USE_ZOOM}")
    print(f"Área: X[{X_MIN:.3f}-{X_MAX:.3f}], Y[{Y_MIN:.3f}-{Y_MAX:.3f}]")
    print(f"Total Simulações: {NUM_SIMULATIONS}")
    print(f"Batch Size: {BATCH_SIZE}")

    num_batches = int(np.ceil(NUM_SIMULATIONS / BATCH_SIZE))
    total_salvo = 0

    # Dimensões da janela de zoom
    width = X_MAX - X_MIN
    height = Y_MAX - Y_MIN

    for b in range(num_batches):
        # 1. Gerar coordenadas normalizadas (0 a 1)
        rx = np.random.rand(BATCH_SIZE)
        ry = np.random.rand(BATCH_SIZE)
        
        # 2. Mapear para a janela de Zoom
        pos_x = X_MIN + (rx * width)
        pos_y = Y_MIN + (ry * height)
        
        # 3. Preparar entrada para o Numba
        posicoes_lote = np.column_stack((pos_x, pos_y))

        # 4. Rodar Simulação (Sem passar a matriz de grid)
        somas = simular_varias_posicoes(posicoes_lote, GRID_SIZE, SPACE_SIZE, TIME_STEPS, DT, G)

        # 5. Salvar Resultados
        df_batch = pd.DataFrame({
            'x': posicoes_lote[:, 0].astype(np.float32),
            'y': posicoes_lote[:, 1].astype(np.float32),
            'soma_total': somas.astype(np.float32)
        })

        arquivo_saida = os.path.join(OUTPUT_DIR, f"lote_{b:05d}.parquet")
        df_batch.to_parquet(arquivo_saida, engine='pyarrow', compression='snappy')
        
        total_salvo += len(df_batch)
        print(f"Lote {b+1}/{num_batches} salvo. ({total_salvo} total)")

    print("\nProcesso Concluído com Sucesso!")