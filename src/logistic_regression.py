"""
logistic_regression.py
======================
Regressão logística escrita de raiz (só numpy), treinada com gradient descent.

Diferenças em relação à regressão linear:
  - a previsão passa por uma "função sigmoide", que transforma qualquer número
    numa probabilidade entre 0 e 1;
  - o custo é a "log-loss" (entropia cruzada), não o erro quadrático;
  - no fim, usamos um LIMIAR (por defeito 0.5) para passar de probabilidade a classe 0/1.

Fórmulas (m = nº de exemplos, w = pesos, b = termo independente):
  z = X @ w + b
  p = sigmoide(z) = 1 / (1 + e^(-z))                 (probabilidade de ser "caro")
  custo:      J = -(1/m) * soma( y*log(p) + (1-y)*log(1-p) )  +  (lambda / 2m) * soma(w^2)
  gradientes: dw = (1/m) * X.T @ (p - y)  +  (lambda / m) * w
              db = (1/m) * soma(p - y)
  passo:      w = w - taxa * dw        b = b - taxa * db
(Repara: os gradientes têm a mesma forma que na linear; só muda o p. O b NÃO é regularizado.)

Desequilíbrio de classes (só ~20% de "caros"):
  peso_classe="balanceado" dá mais importância aos exemplos da classe rara
  (cada exemplo pesa m / (2 * nº da sua classe)), para o modelo não ignorar os caros.
"""

import numpy as np


# ----------------------------------------------------------------------
# Peças matemáticas
# ----------------------------------------------------------------------
def sigmoide(z):
    """1 / (1 + e^-z), escrita de forma estável (sem overflow para z muito grande/pequeno)."""
    z = np.asarray(z, dtype=float)
    saida = np.empty_like(z)
    positivo = z >= 0
    saida[positivo] = 1.0 / (1.0 + np.exp(-z[positivo]))
    exp_z = np.exp(z[~positivo])
    saida[~positivo] = exp_z / (1.0 + exp_z)
    return saida


def log_loss(y_real, prob, eps=1e-12):
    """Entropia cruzada média (quanto menor, melhor)."""
    y_real = np.asarray(y_real, dtype=float)
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_real * np.log(prob) + (1 - y_real) * np.log(1 - prob)))


def accuracy(y_real, y_prev):
    """Fração de previsões certas."""
    return float(np.mean(np.asarray(y_real) == np.asarray(y_prev)))


# ----------------------------------------------------------------------
# Métricas de classificação (Etapa 12)
# ----------------------------------------------------------------------
def matriz_confusao(y_real, y_prev):
    """Devolve (VN, FP, FN, VP): VP = disse "caro" e era; FP = disse "caro" e não era; etc."""
    y_real = np.asarray(y_real).astype(int)
    y_prev = np.asarray(y_prev).astype(int)
    vn = int(np.sum((y_prev == 0) & (y_real == 0)))
    fp = int(np.sum((y_prev == 1) & (y_real == 0)))
    fn = int(np.sum((y_prev == 0) & (y_real == 1)))
    vp = int(np.sum((y_prev == 1) & (y_real == 1)))
    return vn, fp, fn, vp


def precision_recall_f1(y_real, y_prev):
    """precision = VP/(VP+FP) | recall = VP/(VP+FN) | F1 = média harmónica dos dois."""
    vn, fp, fn, vp = matriz_confusao(y_real, y_prev)
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    recall = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _contagens_por_limiar(y_real, prob):
    """
    Para cada limiar distinto (do mais alto para o mais baixo), conta VP e FP acumulados
    (a previsão é "caro" quando prob >= limiar). Os empates entram todos de uma vez.
    """
    y_real = np.asarray(y_real).astype(int)
    prob = np.asarray(prob, dtype=float)
    ordem = np.argsort(-prob, kind="mergesort")
    prob_ord = prob[ordem]
    y_ord = y_real[ordem]
    vp_acum = np.cumsum(y_ord)
    fp_acum = np.cumsum(1 - y_ord)
    ultimo_de_cada = np.r_[np.where(np.diff(prob_ord) != 0)[0], len(prob_ord) - 1]
    return prob_ord[ultimo_de_cada], vp_acum[ultimo_de_cada], fp_acum[ultimo_de_cada]


def curva_roc(y_real, prob):
    """Pontos da curva ROC: (taxa de falsos positivos, taxa de verdadeiros positivos = recall)."""
    limiares, vp, fp = _contagens_por_limiar(y_real, prob)
    n1 = int(np.sum(np.asarray(y_real) == 1))
    n0 = len(y_real) - n1
    tpr = np.r_[0.0, vp / n1]
    fpr = np.r_[0.0, fp / n0]
    return fpr, tpr, np.r_[np.inf, limiares]


def auc_roc(y_real, prob):
    """Área sob a curva ROC (regra dos trapézios). 0.5 = sorte; 1 = perfeito."""
    fpr, tpr, _ = curva_roc(y_real, prob)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def curva_pr(y_real, prob):
    """Pontos da curva precision-recall (do limiar mais alto para o mais baixo)."""
    limiares, vp, fp = _contagens_por_limiar(y_real, prob)
    n1 = int(np.sum(np.asarray(y_real) == 1))
    precision = vp / (vp + fp)
    recall = vp / n1
    return precision, recall, limiares


def average_precision(y_real, prob):
    """Área sob a curva PR (soma por degraus): AP = soma((recall_i - recall_i-1) * precision_i)."""
    precision, recall, _ = curva_pr(y_real, prob)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def tabela_limiares(y_real, prob, limiares=None):
    """Tabela com precision, recall, F1 e accuracy para vários limiares."""
    import pandas as pd

    if limiares is None:
        limiares = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    prob = np.asarray(prob, dtype=float)
    linhas = []
    for lim in limiares:
        prev = (prob >= lim).astype(int)
        p, r, f = precision_recall_f1(y_real, prev)
        linhas.append({"limiar": lim, "precision": round(p, 3), "recall": round(r, 3),
                       "F1": round(f, 3), "accuracy": round(accuracy(y_real, prev), 3),
                       "% previstos caro": round(float(prev.mean()), 3)})
    return pd.DataFrame(linhas)


def melhor_limiar(y_real, prob, criterio="f1", precision_minima=None):
    """
    Procura o limiar que maximiza o F1 (criterio="f1"), a accuracy (criterio="accuracy")
    ou o recall garantindo uma precision mínima (criterio="recall", precision_minima=0.6).
    Deve ser usado SÓ na validação.
    """
    prob = np.asarray(prob, dtype=float)
    melhor, melhor_valor = 0.5, -1.0
    for lim in np.unique(np.round(np.linspace(0.02, 0.95, 94), 2)):
        prev = (prob >= lim).astype(int)
        p, r, f = precision_recall_f1(y_real, prev)
        if criterio == "f1":
            valor = f
        elif criterio == "accuracy":
            valor = accuracy(y_real, prev)
        elif criterio == "recall":
            valor = r if (precision_minima is None or p >= precision_minima) else -1.0
        else:
            raise ValueError("criterio tem de ser 'f1', 'accuracy' ou 'recall'")
        if valor > melhor_valor:
            melhor, melhor_valor = float(lim), valor
    return melhor, float(melhor_valor)


def grafico_matriz_confusao(y_real, y_prev, caminho=None, titulo="Matriz de confusão"):
    import matplotlib.pyplot as plt

    vn, fp, fn, vp = matriz_confusao(y_real, y_prev)
    valores = np.array([[vn, fp], [fn, vp]])
    fig, eixo = plt.subplots(figsize=(4.5, 4))
    eixo.imshow(valores, cmap="Blues")
    eixo.set_xticks([0, 1], ["não caro", "caro"])
    eixo.set_yticks([0, 1], ["não caro", "caro"])
    eixo.set_xlabel("Previsto")
    eixo.set_ylabel("Real")
    eixo.set_title(titulo)
    for i in range(2):
        for j in range(2):
            cor = "white" if valores[i, j] > valores.max() / 2 else "black"
            eixo.text(j, i, str(valores[i, j]), ha="center", va="center", color=cor, fontsize=13)
    fig.tight_layout()
    if caminho is not None:
        fig.savefig(caminho, dpi=150)
    return fig


def grafico_roc_pr(y_real, prob, caminho=None, limiar_marcado=None):
    """Duas figuras lado a lado: curva ROC e curva precision-recall (com o limiar marcado)."""
    import matplotlib.pyplot as plt

    y_real = np.asarray(y_real)
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(10, 4))

    fpr, tpr, _ = curva_roc(y_real, prob)
    e1.plot(fpr, tpr, label=f"AUC = {auc_roc(y_real, prob):.3f}")
    e1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="sorte")
    e1.set_xlabel("Taxa de falsos positivos")
    e1.set_ylabel("Recall (taxa de verdadeiros positivos)")
    e1.set_title("Curva ROC")
    e1.legend(loc="lower right")

    precision, recall, _ = curva_pr(y_real, prob)
    e2.plot(recall, precision, label=f"AP = {average_precision(y_real, prob):.3f}")
    e2.axhline(y_real.mean(), color="k", linestyle="--", linewidth=0.8, label="sorte (prevalência)")
    if limiar_marcado is not None:
        prev = (np.asarray(prob) >= limiar_marcado).astype(int)
        p, r, _ = precision_recall_f1(y_real, prev)
        e2.scatter([r], [p], color="red", zorder=3, label=f"limiar {limiar_marcado:.2f}")
        fp_, tp_, _ = curva_roc(y_real, prob)
        n1 = y_real.sum()
        e1.scatter([np.sum((prev == 1) & (y_real == 0)) / (len(y_real) - n1)], [r], color="red", zorder=3,
                   label=f"limiar {limiar_marcado:.2f}")
        e1.legend(loc="lower right")
    e2.set_xlabel("Recall")
    e2.set_ylabel("Precision")
    e2.set_title("Curva precision-recall")
    e2.legend(loc="upper right")
    fig.tight_layout()
    if caminho is not None:
        fig.savefig(caminho, dpi=150)
    return fig


# ----------------------------------------------------------------------
# O modelo
# ----------------------------------------------------------------------
class RegressaoLogisticaGD:
    """
    Regressão logística com gradient descent (batch ou mini-batch).

    Parâmetros
    ----------
    taxa, n_epocas, tamanho_batch, lambda_l2, tolerancia, paciencia, semente:
        iguais aos da regressão linear.
    peso_classe : None (todos os exemplos pesam o mesmo) ou "balanceado".
    limiar : probabilidade a partir da qual prevemos classe 1 (0.5 por defeito).
    """

    def __init__(self, taxa=0.1, n_epocas=1000, tamanho_batch=None, lambda_l2=0.0,
                 tolerancia=1e-8, paciencia=10, semente=42, peso_classe=None, limiar=0.5):
        if peso_classe not in (None, "balanceado"):
            raise ValueError("peso_classe tem de ser None ou 'balanceado'")
        self.taxa = taxa
        self.n_epocas = n_epocas
        self.tamanho_batch = tamanho_batch
        self.lambda_l2 = lambda_l2
        self.tolerancia = tolerancia
        self.paciencia = paciencia
        self.semente = semente
        self.peso_classe = peso_classe
        self.limiar = limiar

        self.w = None
        self.b = 0.0
        self.historico = {"custo_treino": [], "custo_val": []}
        self.convergiu = False
        self.divergiu = False

    # ---------------- previsões ----------------
    def prever_proba(self, X):
        """Probabilidade de ser classe 1 (entre 0 e 1)."""
        return sigmoide(np.asarray(X, dtype=float) @ self.w + self.b)

    def prever(self, X, limiar=None):
        """Classe 0/1: 1 se a probabilidade for >= limiar."""
        limiar = self.limiar if limiar is None else limiar
        return (self.prever_proba(X) >= limiar).astype(int)

    # ---------------- pesos dos exemplos ----------------
    def _aprender_pesos_classe(self, y):
        """Peso de cada classe, calculado SÓ com o treino: m / (2 * nº da classe)."""
        m = len(y)
        n1 = max(float(np.sum(y == 1)), 1.0)
        n0 = max(float(np.sum(y == 0)), 1.0)
        self.peso_classe_1 = m / (2 * n1)
        self.peso_classe_0 = m / (2 * n0)

    def _pesos_amostra(self, y):
        """Peso de cada exemplo (1 para todos, ou balanceado com os pesos aprendidos no treino)."""
        if self.peso_classe is None:
            return np.ones(len(y))
        return np.where(y == 1, self.peso_classe_1, self.peso_classe_0)

    # ---------------- custo e gradientes ----------------
    def custo(self, X, y, pesos=None):
        """Log-loss (com pesos) + penalização L2."""
        m = len(y)
        prob = np.clip(self.prever_proba(X), 1e-12, 1 - 1e-12)
        pesos = np.ones(m) if pesos is None else pesos
        perda = -(y * np.log(prob) + (1 - y) * np.log(1 - prob))
        parte_l2 = self.lambda_l2 * np.sum(self.w ** 2) / (2 * m)
        return float(np.mean(pesos * perda) + parte_l2)

    def _gradientes(self, X, y, pesos):
        m = len(y)
        erro = (sigmoide(X @ self.w + self.b) - y) * pesos
        dw = (X.T @ erro) / m + (self.lambda_l2 / m) * self.w
        db = np.sum(erro) / m
        return dw, db

    # ---------------- treino ----------------
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X_val is not None:
            X_val = np.asarray(X_val, dtype=float)
            y_val = np.asarray(y_val, dtype=float)

        m, n_colunas = X.shape
        rng = np.random.default_rng(self.semente)
        self._aprender_pesos_classe(y)       # pesos de classe calculados SÓ com o treino
        pesos = self._pesos_amostra(y)
        pesos_val = self._pesos_amostra(y_val) if X_val is not None else None

        self.w = np.zeros(n_colunas)
        self.b = 0.0
        self.historico = {"custo_treino": [], "custo_val": []}
        self.convergiu = False
        self.divergiu = False

        tamanho = m if self.tamanho_batch is None else min(self.tamanho_batch, m)
        sem_melhoria = 0
        custo_anterior = np.inf
        custo_inicial = self.custo(X, y, pesos)   # com pesos a zero ≈ 0.693

        for epoca in range(1, self.n_epocas + 1):
            ordem = rng.permutation(m) if tamanho < m else np.arange(m)

            for inicio in range(0, m, tamanho):
                idx = ordem[inicio:inicio + tamanho]
                dw, db = self._gradientes(X[idx], y[idx], pesos[idx])
                self.w -= self.taxa * dw
                self.b -= self.taxa * db

            custo_t = self.custo(X, y, pesos)
            self.historico["custo_treino"].append(custo_t)
            if X_val is not None:
                self.historico["custo_val"].append(self.custo(X_val, y_val, pesos_val))

            if verbose and (epoca == 1 or epoca % 50 == 0):
                print(f"época {epoca:4d} | custo treino = {custo_t:.6f}")

            # Se o custo ficar muito pior do que no início, a taxa era grande demais
            if not np.isfinite(custo_t) or custo_t > 5 * max(custo_inicial, 1e-12):
                self.divergiu = True
                break

            if abs(custo_anterior - custo_t) < self.tolerancia:
                sem_melhoria += 1
            else:
                sem_melhoria = 0
            custo_anterior = custo_t
            if sem_melhoria >= self.paciencia:
                self.convergiu = True
                break

        return self

    # ---------------- gráfico e coeficientes ----------------
    def grafico_convergencia(self, caminho=None, titulo="Convergência da regressão logística"):
        import matplotlib.pyplot as plt

        fig, eixo = plt.subplots(figsize=(7, 4))
        eixo.plot(self.historico["custo_treino"], label="treino")
        if self.historico["custo_val"]:
            eixo.plot(self.historico["custo_val"], label="validação", linestyle="--")
        eixo.set_xlabel("Época")
        eixo.set_ylabel("Custo (log-loss)")
        eixo.set_title(titulo)
        eixo.legend()
        fig.tight_layout()
        if caminho is not None:
            fig.savefig(caminho, dpi=150)
        return fig

    def tabela_coeficientes(self, nomes_colunas):
        """Cada variável, o peso e o odds ratio (e^peso): >1 aumenta a chance de ser caro, <1 diminui."""
        import pandas as pd

        tabela = pd.DataFrame({"variavel": list(nomes_colunas), "peso": self.w})
        tabela["odds_ratio"] = np.exp(tabela["peso"])
        tabela["abs"] = tabela["peso"].abs()
        tabela = tabela.sort_values("abs", ascending=False).drop(columns="abs")
        return tabela.reset_index(drop=True)


if __name__ == "__main__":
    # Demonstração: dados inventados com fronteira conhecida
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    z = 2 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    y = (rng.random(2000) < sigmoide(z)).astype(int)

    modelo = RegressaoLogisticaGD(taxa=0.5, n_epocas=5000).fit(X, y)
    print("pesos aprendidos:", modelo.w.round(2), "| b:", round(modelo.b, 2), "(esperado ≈ 2, -1.5, 0.5)")
    print("convergiu:", modelo.convergiu, "| accuracy:", round(accuracy(y, modelo.prever(X)), 3))
