"""
experiments.py
==============
Etapa 13: compara estratégias de pré-processamento com validação cruzada (k-fold),
para a REGRESSÃO (preço) e para a CLASSIFICAÇÃO (caro / não caro).

    python src/experiments.py                       -> estudo principal (imputação x outliers), 5 folds
    python src/experiments.py --folds 3             -> mais rápido
    python src/experiments.py --estudo min_noites   -> estudo do limite da estadia mínima
    python src/experiments.py --estudo extras       -> confirma os extras (interações e polinómios)
    python src/experiments.py --estudo todos        -> os três

Sem leakage: só se usam os dados de desenvolvimento (treino + validação = 80%); o TESTE não é tocado.
Em CADA fold, o pré-processamento (medianas, médias, limites de outliers, limiares do "caro")
é aprendido só com o treino desse fold e aplicado ao fold de validação, cujo alvo nunca é alterado.

Nota: o capping só altera o alvo do preço no treino. Como o "caro" é definido com o preço
original, na CLASSIFICAÇÃO o capping dá exatamente o mesmo resultado que "nada".
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data_preparation as dp
import linear_regression as lin
import logistic_regression as lg

PASTA_FIGURAS = Path(__file__).resolve().parent.parent / "figures"

EXTRAS = ["dist_x_lisboa", "poly2"]                 # configuração escolhida nas Etapas 9 e 11
IMPUTACOES = ["simples", "grupos"]
OUTLIERS = ["nada", "capping", "filtrar"]
LIMITES_MIN_NOITES = [None, 90, 30, 7]              # None = sem limite

# Menos exigentes do que no main.py (são muitos treinos); iguais para todas as estratégias
CFG_REG = dict(taxa=0.5, n_epocas=10000, tolerancia=1e-7)
CFG_LOG = dict(taxa=0.5, n_epocas=6000, tolerancia=1e-6)


def treinar_regressao(d):
    """Treina no log-preço e avalia em euros no fold de validação."""
    taxa = CFG_REG["taxa"]
    for _ in range(5):
        modelo = lin.RegressaoLinearGD(taxa=taxa, n_epocas=CFG_REG["n_epocas"], tolerancia=CFG_REG["tolerancia"])
        modelo.fit(d["X_treino"], d["log_treino"])
        if not modelo.divergiu:
            break
        taxa /= 3
    prev_log = modelo.prever(d["X_val"])
    m = lin.todas_metricas(d["preco_val"], np.expm1(prev_log))
    return {"MAE": m["MAE"], "RMSE": m["RMSE"], "R2": m["R2"], "R2_log": lin.r2(d["log_val"], prev_log),
            "convergiu": modelo.convergiu, "épocas": len(modelo.historico["custo_treino"])}


def treinar_logistica(d):
    modelo = lg.RegressaoLogisticaGD(taxa=CFG_LOG["taxa"], n_epocas=CFG_LOG["n_epocas"],
                                     tolerancia=CFG_LOG["tolerancia"])
    modelo.fit(d["X_treino"], d["y_treino_caro"])
    y = d["y_val_caro"]
    prob = modelo.prever_proba(d["X_val"])
    prev = (prob >= 0.5).astype(int)
    p, r, f1 = lg.precision_recall_f1(y, prev)
    return {"AUC": lg.auc_roc(y, prob), "PR_AUC": lg.average_precision(y, prob),
            "log_loss": lg.log_loss(y, prob), "precision@0.5": p, "recall@0.5": r, "F1@0.5": f1,
            "convergiu": modelo.convergiu, "épocas": len(modelo.historico["custo_treino"])}


def correr_estrategia(dev, k, tarefa, imputacao, outliers, nome, extras=EXTRAS):
    linhas = []
    for i, (tr, va) in enumerate(dp.folds_validacao_cruzada(dev, k), start=1):
        d = dp.preparar_fold(tr, va, imputacao, outliers, extras)
        res = treinar_regressao(d) if tarefa == "regressao" else treinar_logistica(d)
        linhas.append({"estrategia": nome, "fold": i, **res})
    return linhas


def resumo(tabela, metricas, baseline, menor_e_melhor):
    """Média ± desvio-padrão por estratégia e comparação com a baseline, fold a fold."""
    saida = []
    base = tabela[tabela["estrategia"] == baseline].set_index("fold")
    for nome, g in tabela.groupby("estrategia", sort=False):
        linha = {"estrategia": nome}
        g = g.set_index("fold")
        for m in metricas:
            linha[m] = f"{g[m].mean():.3f} ± {g[m].std():.3f}"
        m0 = metricas[0]
        dif = g[m0] - base[m0]
        melhor = (dif < 0) if menor_e_melhor else (dif > 0)
        linha[f"dif {m0} vs baseline"] = round(float(dif.mean()), 4)
        linha[f"folds melhores ({m0})"] = f"{int(melhor.sum())}/{len(g)}"
        linha["convergiu"] = f"{int(g['convergiu'].sum())}/{len(g)}"
        saida.append(linha)
    return pd.DataFrame(saida)


def grafico(tabela, metrica, caminho, titulo):
    import matplotlib.pyplot as plt
    g = tabela.groupby("estrategia", sort=False)[metrica].agg(["mean", "std"])
    fig, eixo = plt.subplots(figsize=(8, 4))
    eixo.bar(range(len(g)), g["mean"], yerr=g["std"], capsize=4, color="#4c78a8")
    eixo.set_xticks(range(len(g)), g.index, rotation=30, ha="right")
    eixo.set_ylabel(metrica)
    eixo.set_title(titulo)
    lim = g["mean"].min() - 2 * g["std"].max(), g["mean"].max() + 2 * g["std"].max()
    eixo.set_ylim(lim)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)


def estudo_preprocessamento(dev, k):
    PASTA_FIGURAS.mkdir(exist_ok=True)
    baseline = "simples + nada"

    print("\n=== REGRESSÃO (preço, avaliado em euros) – %d folds ===" % k)
    linhas = []
    for imp in IMPUTACOES:
        for out in OUTLIERS:
            t0 = time.time()
            linhas += correr_estrategia(dev, k, "regressao", imp, out, f"{imp} + {out}")
            print(f"  {imp} + {out} feito ({time.time() - t0:.0f}s)")
    tab_reg = pd.DataFrame(linhas)
    tab_reg.to_csv(PASTA_FIGURAS / "etapa13_regressao_folds.csv", index=False)
    res_reg = resumo(tab_reg, ["MAE", "RMSE", "R2", "R2_log"], baseline, menor_e_melhor=True)
    print(res_reg.to_string(index=False))
    res_reg.to_csv(PASTA_FIGURAS / "etapa13_regressao.csv", index=False)
    grafico(tab_reg, "MAE", PASTA_FIGURAS / "etapa13_regressao_mae.png", "MAE por estratégia (média ± desvio, k-fold)")

    print("\n=== CLASSIFICAÇÃO (caro / não caro) – %d folds ===" % k)
    print("(capping = nada na classificação: o 'caro' usa o preço original; não se treina de novo)")
    linhas = []
    for imp in IMPUTACOES:
        for out in ["nada", "filtrar"]:
            t0 = time.time()
            linhas += correr_estrategia(dev, k, "classificacao", imp, out, f"{imp} + {out}")
            print(f"  {imp} + {out} feito ({time.time() - t0:.0f}s)")
        copia = [dict(l, estrategia=f"{imp} + capping") for l in linhas if l["estrategia"] == f"{imp} + nada"]
        linhas += copia
    tab_clf = pd.DataFrame(linhas)
    ordem = [f"{i} + {o}" for i in IMPUTACOES for o in OUTLIERS]
    tab_clf["estrategia"] = pd.Categorical(tab_clf["estrategia"], ordem, ordered=True)
    tab_clf = tab_clf.sort_values(["estrategia", "fold"]).reset_index(drop=True)
    tab_clf["estrategia"] = tab_clf["estrategia"].astype(str)
    tab_clf.to_csv(PASTA_FIGURAS / "etapa13_classificacao_folds.csv", index=False)
    res_clf = resumo(tab_clf, ["AUC", "PR_AUC", "log_loss", "F1@0.5"], baseline, menor_e_melhor=False)
    print(res_clf.to_string(index=False))
    res_clf.to_csv(PASTA_FIGURAS / "etapa13_classificacao.csv", index=False)
    grafico(tab_clf, "AUC", PASTA_FIGURAS / "etapa13_classificacao_auc.png", "AUC por estratégia (média ± desvio, k-fold)")


def estudo_min_noites(df, k):
    PASTA_FIGURAS.mkdir(exist_ok=True)
    linhas_reg, linhas_clf = [], []
    for limite in LIMITES_MIN_NOITES:
        nome = "sem limite" if limite is None else f"limite {limite}"
        feat = dp.criar_features(df, max_min_noites=limite)
        _, _, _, dev = dp.dividir_dados(feat)
        t0 = time.time()
        linhas_reg += correr_estrategia(dev, k, "regressao", "simples", "nada", nome)
        linhas_clf += correr_estrategia(dev, k, "classificacao", "simples", "nada", nome)
        print(f"  {nome} feito ({time.time() - t0:.0f}s)")
    base = "limite 30"
    tab_reg, tab_clf = pd.DataFrame(linhas_reg), pd.DataFrame(linhas_clf)
    print("\n=== ESTADIA MÍNIMA: regressão ===")
    res = resumo(tab_reg, ["MAE", "RMSE", "R2", "R2_log"], base, True)
    print(res.to_string(index=False)); res.to_csv(PASTA_FIGURAS / "etapa13_min_noites_regressao.csv", index=False)
    print("\n=== ESTADIA MÍNIMA: classificação ===")
    res = resumo(tab_clf, ["AUC", "PR_AUC", "log_loss", "F1@0.5"], base, False)
    print(res.to_string(index=False)); res.to_csv(PASTA_FIGURAS / "etapa13_min_noites_classificacao.csv", index=False)


def estudo_extras(dev, k):
    """Confirma com k-fold se as interações e os polinómios (Etapa 9) compensam."""
    PASTA_FIGURAS.mkdir(exist_ok=True)
    opcoes = [("sem extras", None), ("dist_x_lisboa", ["dist_x_lisboa"]), ("poly2", ["poly2"]),
              ("dist_x_lisboa + poly2", ["dist_x_lisboa", "poly2"])]
    linhas_reg, linhas_clf = [], []
    for nome, extras in opcoes:
        t0 = time.time()
        linhas_reg += correr_estrategia(dev, k, "regressao", "simples", "nada", nome, extras)
        linhas_clf += correr_estrategia(dev, k, "classificacao", "simples", "nada", nome, extras)
        print(f"  {nome} feito ({time.time() - t0:.0f}s)")
    base = "sem extras"
    print("\n=== EXTRAS: regressão ===")
    res = resumo(pd.DataFrame(linhas_reg), ["MAE", "RMSE", "R2", "R2_log"], base, True)
    print(res.to_string(index=False)); res.to_csv(PASTA_FIGURAS / "etapa13_extras_regressao.csv", index=False)
    print("\n=== EXTRAS: classificação ===")
    res = resumo(pd.DataFrame(linhas_clf), ["AUC", "PR_AUC", "log_loss", "F1@0.5"], base, False)
    print(res.to_string(index=False)); res.to_csv(PASTA_FIGURAS / "etapa13_extras_classificacao.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Etapa 13: comparação de estratégias com k-fold.")
    ap.add_argument("--dados", default=None, help="caminho para o listings.csv.gz")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--estudo", choices=["preprocessamento", "min_noites", "extras", "todos"], default="preprocessamento")
    args = ap.parse_args()

    df = dp.carregar_dados(args.dados)
    feat = dp.criar_features(df)
    _, _, _, dev = dp.dividir_dados(feat)     # o teste é descartado aqui: não se usa nesta etapa
    print(f"Dados de desenvolvimento (treino+validação): {len(dev)} anúncios | {args.folds} folds")

    if args.estudo in ("preprocessamento", "todos"):
        estudo_preprocessamento(dev, args.folds)
    if args.estudo in ("min_noites", "todos"):
        estudo_min_noites(df, args.folds)
    if args.estudo in ("extras", "todos"):
        estudo_extras(dev, args.folds)
