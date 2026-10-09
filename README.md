# Projeto Prático 1 – Machine Learning (2026/27)
Aprendizagem supervisionada com dados do Inside Airbnb: prever o preço por noite (regressão linear) e detetar anúncios "caros" (regressão logística). Os modelos estão implementados de raiz com NumPy (gradiente descendente) e validados contra o scikit-learn.

## Dataset
- Fonte: Inside Airbnb – https://insideairbnb.com/get-the-data/
- Região: Lisboa e arredores (inclui Cascais, Sintra, Mafra, Oeiras)
- Ficheiro: `listings.csv.gz`
- Versão/data do scrape: 19 de setembro de 2026 (Inside Airbnb, Lisboa)
- Colocar o ficheiro em `data/raw/listings.csv.gz` (não vai no repositório).

## Estrutura
```
src/
  data_preparation.py   limpeza, features, split 60/20/20, k-fold, pré-processamento, alvo "caro"
  linear_regression.py  regressão linear com GD (batch/mini-batch, L2, gráficos de convergência)
  logistic_regression.py regressão logística com GD + métricas (precision, recall, F1, ROC, PR)
  main.py               corre tudo na validação
  experiments.py        estudos com 5-fold (imputação, outliers, min. noites, extras)
  final_analysis.py     avaliação final no teste, coeficientes, resíduos, erros por grupo
data/raw/               dados originais
figures/                tabelas e gráficos gerados
notebooks/              exploração (missingness, outliers)
```

## Como reproduzir
```
pip install -r requirements.txt
python src/main.py                      # treino + validação
python src/experiments.py --estudo todos   # ou: preprocessamento | min_noites | extras  (opção --folds)
python src/final_analysis.py            # usa o TESTE: correr só no fim, uma vez
```
Para o notebook: abrir `notebooks/` e correr todas as células de cima para baixo (a primeira faz `sys.path.append("../src")`).

## Decisões principais
- Sem leakage: tudo o que se aprende (medianas, médias/desvios, limites de outliers, categorias raras, limiares de "caro", pesos de classe) vem só do treino; no k-fold o pré-processamento é refeito em cada fold.
- Alvo da regressão: log do preço (volta-se a euros com expm1).
- Classe positiva: preço acima do percentil 80 do treino por grupo [tipo de quarto, concelho].
- Limiar de decisão (0,25) escolhido na validação.
- Semente fixa: 42.

## Notas
- `host_response_rate` vem vazia nesta versão do dataset, por isso não é usada.
- Os nomes dos bairros vêm sem acentos (codificação do dataset).
- Versões das bibliotecas: fixar com `pip freeze > requirements.txt` no ambiente usado.
