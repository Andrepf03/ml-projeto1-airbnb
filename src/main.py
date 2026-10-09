"""
main.py
=======
Script principal: corre o projeto de ponta a ponta com os dados reais.

    python src/main.py              -> treina e avalia na VALIDAÇÃO
    python src/main.py --teste      -> avalia também no TESTE (só para o fim!)

Passos:
  1. data_preparation.py   -> limpeza, features, split 60/20/20, matrizes
  2. linear_regression.py  -> prever o preço (alvo = log do preço)
  3. logistic_regression.py-> prever "caro" (preço acima do percentil 80 do grupo)

As escolhas (configurações) estão no topo, para se verem e mudarem facilmente.
Os resultados são guardados em figures/ (tabelas e gráficos).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_preparation as dp
import linear_regression as lin
import logistic_regression as lg

PASTA_FIGURAS = Path(__file__).resolve().parent.parent / "figures"

# ---------------- configurações escolhidas (atualizar com as conclusões das etapas) ----------------
CONFIG_REGRESSAO = dict(imputacao="simples", outliers="nada", extras=["dist_x_lisboa", "poly2"],
                        taxa=0.5, n_epocas=30000, tolerancia=1e-8, lambda_l2=0.0)
CONFIG_LOGISTICA = dict(imputacao="simples", outliers="nada", extras=["dist_x_lisboa", "poly2"],
                        taxa=0.5, n_epocas=20000, tolerancia=1e-7, lambda_l2=0.0, peso_classe=None)


def correr_regressao(cfg, caminho_dados, usar_teste):
    print("\n=== REGRESSÃO LINEAR (preço) ===")
    r = dp.preparar_tudo(cfg["imputacao"], cfg["outliers"], guardar=False,
                         caminho_dados=caminho_dados, extras=cfg["extras"])
    taxa = cfg["taxa"]
    for _ in range(5):   # se divergir, tenta com a taxa a dividir por 3
        modelo = lin.RegressaoLinearGD(taxa=taxa, n_epocas=cfg["n_epocas"], tolerancia=cfg["tolerancia"],
                                       lambda_l2=cfg["lambda_l2"])
        modelo.fit(r["X_treino"], r["log_treino"], r["X_val"], r["log_val"])
        if not modelo.divergiu:
            break
        taxa /= 3
    print(f"taxa={taxa:.4g} | convergiu={modelo.convergiu} | divergiu={modelo.divergiu} | "
          f"épocas={len(modelo.historico['custo_treino'])}")

    linhas = []
    conjuntos = [("validação", r["X_val"], r["preco_val"])]
    if usar_teste:
        conjuntos.append(("teste", r["X_teste"], r["preco_teste"]))
    for nome, X, preco in conjuntos:
        prev = np.expm1(modelo.prever(X))      # voltar de log-preço a euros
        m = lin.todas_metricas(preco, prev)
        linhas.append({"conjunto": nome, **{k: round(v, 3) for k, v in m.items()}})
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))

    PASTA_FIGURAS.mkdir(exist_ok=True)
    modelo.grafico_convergencia(PASTA_FIGURAS / "regressao_convergencia.png", "Regressão linear (log-preço)")
    tabela.to_csv(PASTA_FIGURAS / "regressao_resultados.csv", index=False)
    return modelo, r


def correr_logistica(cfg, caminho_dados, usar_teste):
    print("\n=== REGRESSÃO LOGÍSTICA (caro / não caro) ===")
    r = dp.preparar_tudo(cfg["imputacao"], cfg["outliers"], guardar=False,
                         caminho_dados=caminho_dados, extras=cfg["extras"])
    assert "price" not in r["X_treino"].columns and "log_price" not in r["X_treino"].columns

    modelo = lg.RegressaoLogisticaGD(taxa=cfg["taxa"], n_epocas=cfg["n_epocas"], tolerancia=cfg["tolerancia"],
                                     lambda_l2=cfg["lambda_l2"], peso_classe=cfg["peso_classe"])
    modelo.fit(r["X_treino"], r["y_treino_caro"], r["X_val"], r["y_val_caro"])
    print(f"convergiu={modelo.convergiu} | divergiu={modelo.divergiu} | "
          f"épocas={len(modelo.historico['custo_treino'])}")

    y_val = r["y_val_caro"]
    prob_val = modelo.prever_proba(r["X_val"])

    # O limiar escolhe-se SÓ na validação (o teste não entra na escolha)
    limiar_f1, _ = lg.melhor_limiar(y_val, prob_val, "f1")
    print(f"limiar 0.50 (por defeito) vs limiar que maximiza o F1 na validação: {limiar_f1:.2f}")

    def linha(nome, y, prob, limiar):
        prev = (prob >= limiar).astype(int)
        vn, fp, fn, vp = lg.matriz_confusao(y, prev)
        p, rec, f1 = lg.precision_recall_f1(y, prev)
        return {"conjunto": nome, "limiar": round(limiar, 2),
                "accuracy": round(lg.accuracy(y, prev), 3), "baseline": round(1 - float(np.mean(y)), 3),
                "precision": round(p, 3), "recall": round(rec, 3), "F1": round(f1, 3),
                "AUC": round(lg.auc_roc(y, prob), 3), "PR-AUC": round(lg.average_precision(y, prob), 3),
                "log_loss": round(lg.log_loss(y, prob), 3), "VN": vn, "FP": fp, "FN": fn, "VP": vp}

    linhas = [linha("validação", y_val, prob_val, 0.5), linha("validação", y_val, prob_val, limiar_f1)]
    if usar_teste:
        y_te = r["y_teste_caro"]
        prob_te = modelo.prever_proba(r["X_teste"])
        linhas += [linha("teste", y_te, prob_te, 0.5), linha("teste", y_te, prob_te, limiar_f1)]
    tabela = pd.DataFrame(linhas)
    print(tabela.to_string(index=False))

    print("\nPrecision / recall / F1 para vários limiares (validação):")
    tab_lim = lg.tabela_limiares(y_val, prob_val)
    print(tab_lim.to_string(index=False))

    PASTA_FIGURAS.mkdir(exist_ok=True)
    modelo.grafico_convergencia(PASTA_FIGURAS / "logistica_convergencia.png", "Regressão logística")
    lg.grafico_matriz_confusao(y_val, (prob_val >= 0.5).astype(int), PASTA_FIGURAS / "logistica_confusao_limiar050.png",
                               "Validação, limiar 0.50")
    lg.grafico_matriz_confusao(y_val, (prob_val >= limiar_f1).astype(int), PASTA_FIGURAS / "logistica_confusao_limiar_f1.png",
                               f"Validação, limiar {limiar_f1:.2f}")
    lg.grafico_roc_pr(y_val, prob_val, PASTA_FIGURAS / "logistica_roc_pr.png", limiar_marcado=limiar_f1)
    tabela.to_csv(PASTA_FIGURAS / "logistica_resultados.csv", index=False)
    tab_lim.to_csv(PASTA_FIGURAS / "logistica_limiares.csv", index=False)
    return modelo, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Corre o projeto de ponta a ponta.")
    ap.add_argument("--dados", default=None, help="caminho para o listings.csv.gz (por defeito: data/raw/)")
    ap.add_argument("--teste", action="store_true", help="avalia também no conjunto de teste (usar só no fim)")
    args = ap.parse_args()

    correr_regressao(CONFIG_REGRESSAO, args.dados, args.teste)
    correr_logistica(CONFIG_LOGISTICA, args.dados, args.teste)
