"""
linear_regression.py
====================
Regressão linear escrita de raiz (só numpy), treinada com gradient descent.

O que este módulo faz:
  - treino em BATCH (usa todos os dados em cada passo) ou MINI-BATCH
    (usa um bocadinho de cada vez);
  - regularização L2 (ridge) opcional, com o parâmetro lambda_l2;
  - guarda o custo de treino (e de validação) em cada época, para o gráfico
    de convergência;
  - métricas: MAE, RMSE e R2.

Ideia central (a "bola a rolar pela colina"):
  O custo J mede o quão mal o modelo está a prever. Em cada passo, calculamos
  para que lado J desce mais depressa (o gradiente) e damos um passo pequeno
  nesse sentido. O tamanho do passo é a learning rate (taxa de aprendizagem).

Fórmulas (m = nº de exemplos, w = pesos, b = termo independente):
  previsão:   y_prev = X @ w + b
  custo:      J = (1 / 2m) * soma((y_prev - y)^2)  +  (lambda / 2m) * soma(w^2)
  gradientes: dw = (1 / m) * X.T @ (y_prev - y)  +  (lambda / m) * w
              db = (1 / m) * soma(y_prev - y)
  passo:      w = w - taxa * dw        b = b - taxa * db
  (o termo independente b NÃO é regularizado)
"""

import numpy as np


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def mae(y_real, y_prev):
    """Erro absoluto médio: em média, quanto erramos (nas unidades do alvo)."""
    return float(np.mean(np.abs(np.asarray(y_real) - np.asarray(y_prev))))


def rmse(y_real, y_prev):
    """Raiz do erro quadrático médio: como o MAE, mas castiga mais os erros grandes."""
    diferenca = np.asarray(y_real) - np.asarray(y_prev)
    return float(np.sqrt(np.mean(diferenca ** 2)))


def r2(y_real, y_prev):
    """R2: fração da variação do alvo que o modelo explica (1 = perfeito, 0 = tão mau como prever a média)."""
    y_real = np.asarray(y_real, dtype=float)
    y_prev = np.asarray(y_prev, dtype=float)
    soma_residuos = np.sum((y_real - y_prev) ** 2)
    soma_total = np.sum((y_real - y_real.mean()) ** 2)
    return float(1 - soma_residuos / soma_total)


def todas_metricas(y_real, y_prev):
    """Devolve um dicionário com as três métricas."""
    return {"MAE": mae(y_real, y_prev), "RMSE": rmse(y_real, y_prev), "R2": r2(y_real, y_prev)}


# ----------------------------------------------------------------------
# O modelo
# ----------------------------------------------------------------------
class RegressaoLinearGD:
    """
    Regressão linear com gradient descent.

    Parâmetros
    ----------
    taxa : learning rate (tamanho do passo). Demasiado grande -> o custo explode;
           demasiado pequena -> demora uma eternidade.
    n_epocas : nº máximo de épocas (1 época = passar uma vez por todos os dados).
    tamanho_batch : None = batch (todos os dados por passo);
                    um número (ex.: 64) = mini-batch.
    lambda_l2 : força da regularização L2 (0 = sem regularização).
    tolerancia : se o custo melhorar menos do que isto durante `paciencia`
                 épocas seguidas, paramos (convergiu).
    paciencia : quantas épocas seguidas "sem melhorar" antes de parar.
    semente : para o baralhar dos mini-batches ser sempre igual.
    """

    def __init__(self, taxa=0.1, n_epocas=500, tamanho_batch=None,
                 lambda_l2=0.0, tolerancia=1e-8, paciencia=10, semente=42):
        self.taxa = taxa
        self.n_epocas = n_epocas
        self.tamanho_batch = tamanho_batch
        self.lambda_l2 = lambda_l2
        self.tolerancia = tolerancia
        self.paciencia = paciencia
        self.semente = semente

        self.w = None          # pesos (um por coluna de X)
        self.b = 0.0           # termo independente
        self.historico = {"custo_treino": [], "custo_val": []}
        self.convergiu = False
        self.divergiu = False

    # ---------------- peças matemáticas ----------------
    def prever(self, X):
        """y_prev = X @ w + b"""
        return np.asarray(X, dtype=float) @ self.w + self.b

    def custo(self, X, y):
        """J = (1/2m) * soma((y_prev - y)^2) + (lambda/2m) * soma(w^2)"""
        m = len(y)
        erro = self.prever(X) - y
        parte_erro = np.sum(erro ** 2) / (2 * m)
        parte_l2 = self.lambda_l2 * np.sum(self.w ** 2) / (2 * m)
        return float(parte_erro + parte_l2)

    def _gradientes(self, X, y):
        """Devolve (dw, db) para os dados X, y que recebe."""
        m = len(y)
        erro = X @ self.w + self.b - y
        dw = (X.T @ erro) / m + (self.lambda_l2 / m) * self.w
        db = np.sum(erro) / m
        return dw, db

    # ---------------- treino ----------------
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Treina o modelo. X_val e y_val são opcionais (só servem para o gráfico)."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)

        m, n_colunas = X.shape
        rng = np.random.default_rng(self.semente)

        # Começamos com pesos a zero (para o problema ser convexo, qualquer início serve)
        self.w = np.zeros(n_colunas)
        self.b = 0.0
        self.historico = {"custo_treino": [], "custo_val": []}
        self.convergiu = False
        self.divergiu = False

        tamanho = m if self.tamanho_batch is None else min(self.tamanho_batch, m)
        sem_melhoria = 0
        custo_inicial = self.custo(X, y)     # custo com os pesos a zero (ponto de partida)
        custo_anterior = np.inf

        for epoca in range(1, self.n_epocas + 1):
            # Mini-batch: baralhar os dados em cada época; batch: ordem fixa
            if tamanho < m:
                ordem = rng.permutation(m)
            else:
                ordem = np.arange(m)

            for inicio in range(0, m, tamanho):
                indices = ordem[inicio:inicio + tamanho]
                dw, db = self._gradientes(X[indices], y[indices])
                self.w -= self.taxa * dw
                self.b -= self.taxa * db

            # Registar o custo no fim da época (sobre TODOS os dados de treino)
            custo_t = self.custo(X, y)
            self.historico["custo_treino"].append(custo_t)
            if X_val is not None:
                self.historico["custo_val"].append(self.custo(X_val, y_val))

            if verbose and (epoca == 1 or epoca % 50 == 0):
                print(f"época {epoca:4d} | custo treino = {custo_t:.6f}")

            # Se o custo virou NaN/infinito ou ficou muitíssimo pior do que no início,
            # a learning rate era grande demais (o modelo "saltou" para fora da colina)
            if not np.isfinite(custo_t) or custo_t > 1000 * max(custo_inicial, 1e-12):
                self.divergiu = True
                break

            # Convergência: o custo quase não mexe há `paciencia` épocas seguidas
            if abs(custo_anterior - custo_t) < self.tolerancia:
                sem_melhoria += 1
            else:
                sem_melhoria = 0
            custo_anterior = custo_t
            if sem_melhoria >= self.paciencia:
                self.convergiu = True
                break

        return self

    # ---------------- gráfico ----------------
    def grafico_convergencia(self, caminho=None, titulo="Convergência do gradient descent"):
        """Desenha o custo por época (treino e, se existir, validação)."""
        import matplotlib.pyplot as plt

        fig, eixo = plt.subplots(figsize=(7, 4))
        eixo.plot(self.historico["custo_treino"], label="treino")
        if self.historico["custo_val"]:
            eixo.plot(self.historico["custo_val"], label="validação", linestyle="--")
        eixo.set_xlabel("Época")
        eixo.set_ylabel("Custo J")
        eixo.set_title(titulo)
        eixo.set_yscale("log")
        eixo.legend()
        fig.tight_layout()
        if caminho is not None:
            fig.savefig(caminho, dpi=150)
        return fig

    # ---------------- ajuda para o relatório ----------------
    def tabela_coeficientes(self, nomes_colunas):
        """Tabela com cada variável e o seu peso, ordenada do maior para o menor valor absoluto."""
        import pandas as pd

        tabela = pd.DataFrame({"variavel": list(nomes_colunas), "peso": self.w})
        tabela["abs"] = tabela["peso"].abs()
        tabela = tabela.sort_values("abs", ascending=False).drop(columns="abs")
        return tabela.reset_index(drop=True)


if __name__ == "__main__":
    # Mini-demonstração com dados inventados: o modelo deve descobrir w = [3, -2] e b = 5
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + rng.normal(0, 0.1, size=500)

    modelo = RegressaoLinearGD(taxa=0.1, n_epocas=1000).fit(X, y)
    print("pesos aprendidos:", modelo.w.round(3), "| b:", round(modelo.b, 3))
    print("convergiu:", modelo.convergiu, "| épocas:", len(modelo.historico["custo_treino"]))