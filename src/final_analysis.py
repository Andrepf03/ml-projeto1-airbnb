"""
final_analysis.py
=================
Etapa 14: modelos finais, avaliação no TESTE e interpretação.

    python src/final_analysis.py

ATENÇÃO: este script usa o conjunto de TESTE. Só deve ser corrido no fim, depois de todas
as escolhas estarem fechadas (pré-processamento, extras, regularização, limiar).

O que faz:
  1. Treina os modelos finais com treino + validação (80% dos dados), com o pré-processamento
     aprendido só nesses dados, e avalia UMA vez no teste (20%).
  2. Regressão: métricas, tabela de coeficientes, gráficos de resíduos, erro por tipo de quarto,
     concelho e bairro.
  3. Classificação: métricas ao limiar escolhido na validação, matriz de confusão, ROC/PR,
     tabela de coeficientes (odds ratios) e erros por tipo de quarto e concelho.

Tudo é guardado em figures/ (ficheiros etapa14_*).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import data_preparation as dp
import linear_regression as lin
import logistic_regression as lg
import main as cfg_modulo

PASTA_FIGURAS = Path(__file__).resolve().parent.parent / "figures"

LIMIAR_FINAL = 0.25     # limiar que maximizou o F1 na validação (Etapa 12)
MIN_GRUPO = 30          # grupos com menos anúncios no teste não entram nas tabelas de erro


# ----------------------------------------------------------------------
# Treino final
# ----------------------------------------------------------------------
def treinar_regressao_final(d, c):
    taxa = c["taxa"]
    for _ in range(5):
        modelo = lin.RegressaoLinearGD(taxa=taxa, n_epocas=c["n_epocas"], tolerancia=c["tolerancia"],
                                       lambda_l2=c["lambda_l2"])
        modelo.fit(d["X_treino"], d["log_treino"])
        if not modelo.divergiu:
            break
        taxa /= 3
    print(f"  regressão: taxa={taxa:.4g} | convergiu={modelo.convergiu} | "
          f"épocas={len(modelo.historico['custo_treino'])}")
    return modelo


def treinar_logistica_final(d, c):
    modelo = lg.RegressaoLogisticaGD(taxa=c["taxa"], n_epocas=c["n_epocas"], tolerancia=c["tolerancia"],
                                     lambda_l2=c["lambda_l2"], peso_classe=c["peso_classe"])
    modelo.fit(d["X_treino"], d["y_treino_caro"])
    print(f"  logística: convergiu={modelo.convergiu} | divergiu={modelo.divergiu} | "
          f"épocas={len(modelo.historico['custo_treino'])}")
    return modelo


# ----------------------------------------------------------------------
# Tabelas de coeficientes
# ----------------------------------------------------------------------
def tipo_variavel(nome):
    if "=" in nome:
        return "categoria (vs referência)"
    if "^" in nome or nome in ("em_lisboa", "dist_x_lisboa", "acc_x_privado"):
        return "extra (interação/polinómio)"
    if nome in dp.COLUNAS_BINARIAS:
        return "0/1"
    return "numérica (por +1 desvio-padrão)"


def coef_regressao(modelo, nomes):
    t = pd.DataFrame({"variavel": list(nomes), "peso_log": modelo.w})
    t["efeito_%"] = (np.exp(t["peso_log"]) - 1) * 100       # efeito no preço (multiplicativo)
    t["tipo"] = t["variavel"].map(tipo_variavel)
    t["abs"] = t["peso_log"].abs()
    return t.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def coef_logistica(modelo, nomes):
    t = modelo.tabela_coeficientes(nomes)
    t["tipo"] = t["variavel"].map(tipo_variavel)
    return t


# ----------------------------------------------------------------------
# Análise de erros
# ----------------------------------------------------------------------
def erros_regressao_por_grupo(tabela, real, prev, coluna):
    """MAE (€), mediana do erro absoluto e viés mediano (%) por grupo. Viés > 0: o modelo sobrestima."""
    d = pd.DataFrame({"grupo": tabela[coluna].values, "real": real.values, "prev": prev})
    d["erro_abs"] = (d["prev"] - d["real"]).abs()
    d["erro_rel_%"] = (d["prev"] - d["real"]) / d["real"] * 100
    g = d.groupby("grupo").agg(n=("real", "size"), preco_mediano=("real", "median"),
                               MAE=("erro_abs", "mean"), erro_abs_mediano=("erro_abs", "median"),
                               vies_mediano_pct=("erro_rel_%", "median"))
    g = g[g["n"] >= MIN_GRUPO].sort_values("MAE", ascending=False)
    return g.round(2).reset_index().rename(columns={"grupo": coluna})


def erros_classificacao_por_grupo(tabela, y, prev, coluna):
    d = pd.DataFrame({"grupo": tabela[coluna].values, "y": np.asarray(y), "p": np.asarray(prev)})
    linhas = []
    for nome, g in d.groupby("grupo"):
        if len(g) < MIN_GRUPO:
            continue
        p, r, f1 = lg.precision_recall_f1(g["y"], g["p"])
        linhas.append({coluna: nome, "n": len(g), "% caros reais": round(100 * g["y"].mean(), 1),
                       "% previstos caro": round(100 * g["p"].mean(), 1),
                       "precision": round(p, 3), "recall": round(r, 3), "F1": round(f1, 3)})
    return pd.DataFrame(linhas).sort_values("n", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# Gráficos
# ----------------------------------------------------------------------
def graficos_residuos(log_real, log_prev, preco_real, preco_prev, caminho):
    import matplotlib.pyplot as plt

    res = log_real - log_prev
    fig, eixos = plt.subplots(1, 3, figsize=(15, 4.2))

    eixos[0].scatter(log_prev, res, s=4, alpha=0.3)
    eixos[0].axhline(0, color="red", linewidth=1)
    eixos[0].set_xlabel("log(1+preço) previsto")
    eixos[0].set_ylabel("Resíduo (real − previsto, em log)")
    eixos[0].set_title("Resíduos vs previsto (teste)")

    eixos[1].hist(res, bins=60, color="#4c78a8")
    eixos[1].axvline(0, color="red", linewidth=1)
    eixos[1].set_xlabel("Resíduo (log)")
    eixos[1].set_ylabel("Nº de anúncios")
    eixos[1].set_title("Distribuição dos resíduos")

    lim = [min(preco_real.min(), preco_prev.min()) + 1, np.percentile(preco_real, 99.5)]
    eixos[2].scatter(preco_prev, preco_real, s=4, alpha=0.3)
    eixos[2].plot(lim, lim, color="red", linewidth=1)
    eixos[2].set_xscale("log"); eixos[2].set_yscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for eixo in (eixos[2].xaxis, eixos[2].yaxis):
        eixo.set_minor_formatter(NullFormatter())
        eixo.set_major_formatter(ScalarFormatter())
    eixos[2].set_xlim(lim); eixos[2].set_ylim(lim)
    eixos[2].set_xlabel("Preço previsto (€, escala log)")
    eixos[2].set_ylabel("Preço real (€, escala log)")
    eixos[2].set_title("Real vs previsto")

    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def grafico_erro_grupos(tab_tipo, tab_concelho, caminho):
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, 2, figsize=(12, 4.2))
    for eixo, tab, col, titulo in [(eixos[0], tab_tipo, "room_type", "MAE por tipo de quarto"),
                                   (eixos[1], tab_concelho, "concelho", "MAE por concelho")]:
        t = tab.sort_values("MAE")
        eixo.barh(t[col].astype(str), t["MAE"], color="#4c78a8")
        eixo.set_xlabel("MAE (€)")
        eixo.set_title(titulo)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


# ----------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------
def main(caminho_dados=None):
    PASTA_FIGURAS.mkdir(exist_ok=True)
    print("ATENÇÃO: a usar o conjunto de TESTE (só deve acontecer no fim do projeto).\n")

    df = dp.carregar_dados(caminho_dados)
    feat = dp.criar_features(df)
    treino, validacao, teste, dev = dp.dividir_dados(feat)
    print(f"Treino final (treino+validação): {len(dev)} anúncios | Teste: {len(teste)} anúncios")

    cr, cl = cfg_modulo.CONFIG_REGRESSAO, cfg_modulo.CONFIG_LOGISTICA

    # Pré-processamento aprendido em dev (treino+validação), aplicado ao teste
    d_reg = dp.preparar_fold(dev, teste, cr["imputacao"], cr["outliers"], cr["extras"])
    d_log = dp.preparar_fold(dev, teste, cl["imputacao"], cl["outliers"], cl["extras"])

    print("\nA treinar os modelos finais...")
    reg = treinar_regressao_final(d_reg, cr)
    log = treinar_logistica_final(d_log, cl)

    # ================= REGRESSÃO =================
    print("\n=== REGRESSÃO LINEAR: avaliação no teste ===")
    X_te = d_reg["X_val"]
    teste_r = teste.loc[X_te.index]
    log_prev = reg.prever(X_te)
    preco_prev = np.expm1(log_prev)
    preco_real = d_reg["preco_val"]
    log_real = d_reg["log_val"]

    metricas = lin.todas_metricas(preco_real, preco_prev)
    baseline = np.expm1(d_reg["log_treino"].median())       # prever sempre a mediana do treino
    linha = {"MAE": round(metricas["MAE"], 2), "RMSE": round(metricas["RMSE"], 2), "R2": round(metricas["R2"], 3),
             "R2_log": round(lin.r2(log_real, log_prev), 3),
             "MAE_baseline_mediana": round(lin.mae(preco_real, np.full(len(preco_real), baseline)), 2),
             "n_teste": len(preco_real)}
    tab_res = pd.DataFrame([linha])
    print(tab_res.to_string(index=False))
    tab_res.to_csv(PASTA_FIGURAS / "etapa14_teste_regressao.csv", index=False)

    coef_r = coef_regressao(reg, X_te.columns)
    coef_r.to_csv(PASTA_FIGURAS / "etapa14_coeficientes_regressao.csv", index=False)
    print("\nCoeficientes da regressão (log-preço; efeito_% = efeito multiplicativo no preço) – top 20:")
    print(coef_r.head(20).round(3).to_string(index=False))

    graficos_residuos(log_real.values, log_prev, preco_real.values, preco_prev,
                      PASTA_FIGURAS / "etapa14_residuos.png")

    tab_tipo = erros_regressao_por_grupo(teste_r, preco_real, preco_prev, "room_type")
    tab_conc = erros_regressao_por_grupo(teste_r, preco_real, preco_prev, "concelho")
    tab_bairro = erros_regressao_por_grupo(teste_r, preco_real, preco_prev, "neighbourhood_cleansed")
    for nome, t in [("tipo", tab_tipo), ("concelho", tab_conc), ("bairro", tab_bairro)]:
        t.to_csv(PASTA_FIGURAS / f"etapa14_erro_regressao_{nome}.csv", index=False)
    print("\nErro por tipo de quarto:"); print(tab_tipo.to_string(index=False))
    print("\nErro por concelho (n >= %d):" % MIN_GRUPO); print(tab_conc.to_string(index=False))
    print("\nBairros com MAE mais alto (n >= %d):" % MIN_GRUPO); print(tab_bairro.head(8).to_string(index=False))
    print("\nBairros com MAE mais baixo:"); print(tab_bairro.tail(5).to_string(index=False))
    grafico_erro_grupos(tab_tipo, tab_conc, PASTA_FIGURAS / "etapa14_erro_grupos.png")

    # ================= CLASSIFICAÇÃO =================
    print("\n=== REGRESSÃO LOGÍSTICA: avaliação no teste ===")
    X_te_l = d_log["X_val"]
    teste_l = teste.loc[X_te_l.index]
    y = d_log["y_val_caro"]
    prob = log.prever_proba(X_te_l)

    linhas = []
    for lim in (0.5, LIMIAR_FINAL):
        prev = (prob >= lim).astype(int)
        vn, fp, fn, vp = lg.matriz_confusao(y, prev)
        p, r, f1 = lg.precision_recall_f1(y, prev)
        linhas.append({"limiar": lim, "accuracy": round(lg.accuracy(y, prev), 3),
                       "baseline": round(1 - float(y.mean()), 3), "precision": round(p, 3), "recall": round(r, 3),
                       "F1": round(f1, 3), "AUC": round(lg.auc_roc(y, prob), 3),
                       "PR-AUC": round(lg.average_precision(y, prob), 3), "log_loss": round(lg.log_loss(y, prob), 3),
                       "VN": vn, "FP": fp, "FN": fn, "VP": vp})
    tab_cls = pd.DataFrame(linhas)
    print(tab_cls.to_string(index=False))
    tab_cls.to_csv(PASTA_FIGURAS / "etapa14_teste_classificacao.csv", index=False)

    prev_final = (prob >= LIMIAR_FINAL).astype(int)
    lg.grafico_matriz_confusao(y, prev_final, PASTA_FIGURAS / "etapa14_confusao_teste.png",
                               f"Teste, limiar {LIMIAR_FINAL:.2f}")
    lg.grafico_roc_pr(y, prob, PASTA_FIGURAS / "etapa14_roc_pr_teste.png", limiar_marcado=LIMIAR_FINAL)

    coef_l = coef_logistica(log, X_te_l.columns)
    coef_l.to_csv(PASTA_FIGURAS / "etapa14_coeficientes_logistica.csv", index=False)
    print("\nCoeficientes da logística (odds_ratio > 1 aumenta a chance de ser caro) – top 20:")
    print(coef_l.head(20).round(3).to_string(index=False))

    e_tipo = erros_classificacao_por_grupo(teste_l, y, prev_final, "room_type")
    e_conc = erros_classificacao_por_grupo(teste_l, y, prev_final, "concelho")
    e_tipo.to_csv(PASTA_FIGURAS / "etapa14_erro_classificacao_tipo.csv", index=False)
    e_conc.to_csv(PASTA_FIGURAS / "etapa14_erro_classificacao_concelho.csv", index=False)
    print(f"\nErros da classificação por tipo de quarto (limiar {LIMIAR_FINAL}):"); print(e_tipo.to_string(index=False))
    print(f"\nErros da classificação por concelho (n >= {MIN_GRUPO}):"); print(e_conc.to_string(index=False))

    print("\nFicheiros guardados em figures/ (etapa14_*).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etapa 14: avaliação final no teste e interpretação.")
    ap.add_argument("--dados", default=None, help="caminho para o listings.csv.gz")
    main(ap.parse_args().dados)
