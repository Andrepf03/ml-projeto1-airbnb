"""
data_preparation.py
===================
Preparação dos dados do Projeto Prático 1 (Inside Airbnb, Lisboa).

Como se evita o data leakage (fuga de informação):
  1) criar_features()            -> transformações que NÃO aprendem nada dos dados
                                    (seguras de fazer antes de dividir os dados)
  2) dividir_dados()             -> separa treino / validação / teste
  3) PreProcessador.fit()        -> aprende medianas, médias, limites, categorias...
                                    SÓ com o treino
  4) PreProcessador.transformar()-> aplica o que foi aprendido a qualquer conjunto

Uso rápido (a partir da pasta do projeto):
    python src/data_preparation.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

# ----------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------
# Pasta do projeto = uma pasta acima de src/ (assim os caminhos funcionam
# venhas de onde vieres: notebook, terminal, etc.)
RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_DADOS = RAIZ / "data" / "raw" / "listings.csv.gz"
PASTA_PROCESSADOS = RAIZ / "data" / "processed"

SEMENTE = 42            # garante que a divisão é sempre a mesma
MIN_CATEGORIA = 50      # categorias com menos anúncios (no treino) passam a "Outros"
MAX_MIN_NOITES = 30     # estadia mínima limitada a 30 noites (Etapa 4)
CENTRO_LAT = 38.7075    # ~Praça do Comércio, usada como "centro" de Lisboa
CENTRO_LON = -9.1365

# Colunas que lemos do ficheiro original (as da Etapa 1, mais o concelho)
COLUNAS_USADAS = [
    "price", "room_type", "property_type", "neighbourhood_cleansed",
    "neighbourhood_group_cleansed", "latitude", "longitude", "accommodates",
    "bedrooms", "beds", "bathrooms", "bathrooms_text", "amenities",
    "minimum_nights", "host_is_superhost", "number_of_reviews",
    "review_scores_rating", "availability_365",
]

# Variáveis que entram no modelo
COLUNAS_NUMERICAS = [
    "accommodates", "bedrooms", "beds", "bathrooms", "n_amenities",
    "min_noites", "log_reviews", "review_scores_rating", "availability_365",
    "dist_centro_km",
]
COLUNAS_BINARIAS = ["superhost", "sem_reviews", "banho_partilhado"]
COLUNAS_CATEGORICAS = ["room_type", "property_type", "neighbourhood_cleansed"]

# Colunas cujos buracos, na imputação "por grupos", se preenchem pela capacidade
COLUNAS_POR_CAPACIDADE = ["bedrooms", "beds", "bathrooms"]

# ----- Extras da Etapa 9 (interações e polinómios) -----
# Escolhe-se com o parametro `extras` do PreProcessador. Por defeito não há extras.
#   "dist_x_lisboa" : distância ao centro x "está no concelho de Lisboa"
#                     (a distância tem efeito diferente dentro e fora de Lisboa)
#   "acc_x_privado" : capacidade x "quarto privado"
#                     (a capacidade pesa de forma diferente em casas inteiras e em quartos)
#   "poly2"         : quadrados das variáveis em COLUNAS_POLY
#   "poly3"         : quadrados e cubos das variáveis em COLUNAS_POLY
EXTRAS_VALIDOS = ("dist_x_lisboa", "acc_x_privado", "poly2", "poly3")
COLUNAS_POLY = ["accommodates", "bedrooms", "n_amenities", "dist_centro_km"]


# ----------------------------------------------------------------------
# 1) Carregar e limpar o preço (alvo)
# ----------------------------------------------------------------------
def carregar_dados(caminho=None):
    """Lê o ficheiro, converte o preço para número e remove anúncios sem preço."""
    if caminho is None:
        caminho = CAMINHO_DADOS

    df = pd.read_csv(caminho, usecols=COLUNAS_USADAS)

    # O preço vem como texto ("$1,200.00"): tiramos o "$" e a "," e convertemos
    if not pd.api.types.is_numeric_dtype(df["price"]):
        df["price"] = df["price"].str.replace("$", "", regex=False)
        df["price"] = df["price"].str.replace(",", "", regex=False)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # Sem preço não há alvo: removemos (não faz sentido imputar o alvo)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]

    # log(1 + preço): reduz a assimetria
    df["log_price"] = np.log1p(df["price"])
    return df


# ----------------------------------------------------------------------
# 2) Features que NÃO aprendem nada dos dados (seguras antes do split)
# ----------------------------------------------------------------------
def _contar_amenities(texto):
    """Conta quantas comodidades há no texto (que é uma lista em formato JSON)."""
    try:
        return len(json.loads(texto))
    except Exception:
        return np.nan


def distancia_centro_km(lat, lon):
    """Distância em linha reta (km) ao centro de Lisboa (fórmula haversine)."""
    lat0 = np.radians(CENTRO_LAT)
    lon0 = np.radians(CENTRO_LON)
    lat = np.radians(lat)
    lon = np.radians(lon)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin((lon - lon0) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def criar_features(df, max_min_noites=MAX_MIN_NOITES):
    """Cria as variáveis novas. Cada anúncio é tratado sozinho, por isso não há leakage.
    max_min_noites: limite da estadia mínima (None = sem limite; serve para o estudo da Etapa 13)."""
    feat = df.copy()

    # Nº de comodidades
    feat["n_amenities"] = feat["amenities"].apply(_contar_amenities)

    # Casas de banho: recuperar do texto onde o número falta
    texto = feat["bathrooms_text"].str.lower()
    numero = texto.str.extract(r"(\d+\.?\d*)")[0].astype(float)
    meio = texto.str.contains("half", na=False).map({True: 0.5, False: np.nan})
    numero = numero.fillna(meio)
    feat["bathrooms"] = feat["bathrooms"].fillna(numero)

    # Casa de banho partilhada (0/1)
    feat["banho_partilhado"] = texto.str.contains("shared", na=False).astype(int)

    # Superhost: "t" -> 1, "f" -> 0 (os poucos em falta ficam para a imputação)
    feat["superhost"] = feat["host_is_superhost"].map({"t": 1, "f": 0})

    # Ausência com significado: anúncio ainda sem reviews
    feat["sem_reviews"] = (feat["number_of_reviews"] == 0).astype(int)
    feat["log_reviews"] = np.log1p(feat["number_of_reviews"])

    # Estadia mínima limitada a 30 noites
    feat["min_noites"] = feat["minimum_nights"].clip(upper=max_min_noites)

    # Distância ao centro
    feat["dist_centro_km"] = distancia_centro_km(feat["latitude"], feat["longitude"])

    # Concelho (não entra no modelo; serve para a análise de erros)
    feat["concelho"] = feat["neighbourhood_group_cleansed"]

    # Está no concelho de Lisboa? (0/1) - só entra no modelo se se pedir "dist_x_lisboa"
    feat["em_lisboa"] = (feat["concelho"] == "Lisboa").astype(int)
    return feat


# ----------------------------------------------------------------------
# 3) Divisão dos dados
# ----------------------------------------------------------------------
def dividir_dados(feat, semente=SEMENTE):
    """Devolve treino (60%), validação (20%), teste (20%) e dev (= treino + validação)."""
    # Primeiro separa-se o teste e guarda-se: não se toca até ao fim
    dev, teste = train_test_split(
        feat, test_size=0.20, random_state=semente, stratify=feat["room_type"]
    )
    # Do resto tira-se a validação (25% de 80% = 20% do total)
    treino, validacao = train_test_split(
        dev, test_size=0.25, random_state=semente, stratify=dev["room_type"]
    )
    return treino, validacao, teste, dev


def folds_validacao_cruzada(dev, k=5, semente=SEMENTE):
    """Gera k pares (treino_do_fold, validacao_do_fold) a partir dos dados de desenvolvimento."""
    kf = KFold(n_splits=k, shuffle=True, random_state=semente)
    for indices_treino, indices_validacao in kf.split(dev):
        yield dev.iloc[indices_treino], dev.iloc[indices_validacao]


# ----------------------------------------------------------------------
# 4) Pré-processamento que aprende (só com o treino)
# ----------------------------------------------------------------------
class PreProcessador:
    """
    fit(treino)        -> aprende tudo o que depende dos dados (SÓ do treino)
    transformar(dados) -> aplica o que foi aprendido e devolve (X, preço, log-preço)

    estrategia_imputacao: "simples" (mediana global) ou "grupos" (mediana por capacidade)
    tratamento_outliers:  "nada", "capping" (limita o alvo) ou "filtrar" (remove do treino)
    """

    def __init__(self, estrategia_imputacao="simples", tratamento_outliers="nada", extras=None):
        if estrategia_imputacao not in ("simples", "grupos"):
            raise ValueError("estrategia_imputacao tem de ser 'simples' ou 'grupos'")
        if tratamento_outliers not in ("nada", "capping", "filtrar"):
            raise ValueError("tratamento_outliers tem de ser 'nada', 'capping' ou 'filtrar'")
        extras = list(extras) if extras else []
        for e in extras:
            if e not in EXTRAS_VALIDOS:
                raise ValueError("extra desconhecido: " + str(e) + ". Opções: " + str(EXTRAS_VALIDOS))
        if "poly2" in extras and "poly3" in extras:
            raise ValueError("Escolhe 'poly2' ou 'poly3', não os dois (o poly3 já inclui o quadrado)")
        self.estrategia_imputacao = estrategia_imputacao
        self.tratamento_outliers = tratamento_outliers
        self.extras = extras
        self.ajustado = False

    # ---------------- aprender com o treino ----------------
    def fit(self, treino):
        # (a) Medianas globais para imputar
        self.mediana_global = {}
        for col in COLUNAS_NUMERICAS:
            self.mediana_global[col] = treino[col].median()
        # A nota das reviews: mediana só entre anúncios que TÊM reviews
        com_reviews = treino[treino["sem_reviews"] == 0]
        self.mediana_global["review_scores_rating"] = com_reviews["review_scores_rating"].median()
        # Superhost: o valor mais frequente (0 ou 1)
        self.moda_superhost = treino["superhost"].mode()[0]

        # (b) Medianas por capacidade (para a imputação por grupos)
        self.mediana_por_capacidade = {}
        for col in COLUNAS_POR_CAPACIDADE:
            self.mediana_por_capacidade[col] = treino.groupby("accommodates")[col].median()

        # (c) Categorias frequentes (só property_type e neighbourhood_cleansed se agrupam)
        self.categorias_frequentes = {}
        for col in ["property_type", "neighbourhood_cleansed"]:
            contagem = treino[col].value_counts()
            self.categorias_frequentes[col] = list(contagem[contagem >= MIN_CATEGORIA].index)

        # (d) Categorias para o one-hot: a mais frequente é a REFERÊNCIA (fica de fora)
        self.referencia = {}
        self.categorias_dummy = {}
        for col in COLUNAS_CATEGORICAS:
            serie = self._agrupar(treino[col], col)
            ordem = serie.value_counts().index.tolist()   # da mais frequente à menos
            self.referencia[col] = ordem[0]
            self.categorias_dummy[col] = ordem[1:]

        # (e) Limites de outliers no log-preço (IQR e MAD)
        lp = treino["log_price"]
        q1 = lp.quantile(0.25)
        q3 = lp.quantile(0.75)
        iqr = q3 - q1
        self.limite_baixo = q1 - 1.5 * iqr
        self.limite_alto = q3 + 1.5 * iqr
        self.mediana_lp = lp.median()
        self.mad_lp = (lp - self.mediana_lp).abs().median()

        # (f) Média e desvio para escalar (calculados depois de imputar o treino)
        treino_imputado = self._imputar(treino)
        self.media = treino_imputado[COLUNAS_NUMERICAS].mean()
        self.desvio = treino_imputado[COLUNAS_NUMERICAS].std().replace(0, 1)

        # (g) Extras (interações e polinómios): também se padronizam com o TREINO
        self.media_extras = pd.Series(dtype=float)
        self.desvio_extras = pd.Series(dtype=float)
        if self.extras:
            extras_treino = self._extras_brutos(self._numericas_escaladas(treino_imputado), treino_imputado)
            extras_treino = pd.DataFrame(extras_treino)
            self.media_extras = extras_treino.mean()
            self.desvio_extras = extras_treino.std().replace(0, 1)

        self.ajustado = True
        return self

    # ---------------- peças auxiliares ----------------
    def _agrupar(self, serie, col):
        """Categorias raras (ou nunca vistas no treino) passam a 'Outros'."""
        if col in self.categorias_frequentes:
            return serie.where(serie.isin(self.categorias_frequentes[col]), "Outros")
        return serie

    def _imputar(self, tabela):
        """Preenche os buracos usando SÓ valores aprendidos no treino."""
        t = tabela.copy()
        for col in COLUNAS_NUMERICAS:
            if self.estrategia_imputacao == "grupos" and col in COLUNAS_POR_CAPACIDADE:
                # mediana do grupo; se o grupo não existir no treino, usa a global
                por_grupo = t["accommodates"].map(self.mediana_por_capacidade[col])
                preenchimento = por_grupo.fillna(self.mediana_global[col])
            else:
                preenchimento = self.mediana_global[col]
            t[col] = t[col].fillna(preenchimento)
        t["superhost"] = t["superhost"].fillna(self.moda_superhost)
        return t

    def _numericas_escaladas(self, tabela_imputada):
        """Numéricas padronizadas com a média e o desvio do TREINO (dicionário nome -> série)."""
        return {col: (tabela_imputada[col] - self.media[col]) / self.desvio[col]
                for col in COLUNAS_NUMERICAS}

    def _extras_brutos(self, numericas, t):
        """
        Constrói as colunas extra ainda por padronizar (dicionário nome -> série).
        Os polinómios usam as variáveis já centradas (padronizadas): assim x e x^2
        ficam menos correlacionados e o gradient descent comporta-se melhor.
        """
        extras = {}
        if "dist_x_lisboa" in self.extras:
            extras["em_lisboa"] = t["em_lisboa"].astype(float)
            extras["dist_x_lisboa"] = numericas["dist_centro_km"] * t["em_lisboa"]
        if "acc_x_privado" in self.extras:
            e_privado = (t["room_type"] == "Private room").astype(float)
            extras["acc_x_privado"] = numericas["accommodates"] * e_privado
        if "poly2" in self.extras or "poly3" in self.extras:
            for col in COLUNAS_POLY:
                extras[col + "^2"] = numericas[col] ** 2
                if "poly3" in self.extras:
                    extras[col + "^3"] = numericas[col] ** 3
        return extras

    def _mascara_outliers(self, tabela):
        """True nos anúncios que o IQR E o MAD consideram extremos (no log-preço)."""
        lp = tabela["log_price"]
        fora_iqr = (lp < self.limite_baixo) | (lp > self.limite_alto)
        if self.mad_lp == 0:
            return pd.Series(False, index=tabela.index)
        z_modificado = 0.6745 * (lp - self.mediana_lp) / self.mad_lp
        fora_mad = z_modificado.abs() > 3.5
        return fora_iqr & fora_mad

    # ---------------- aplicar ----------------
    def transformar(self, tabela, e_treino=False):
        """
        Devolve (X, preco, log_preco).
        Filtrar/capping aplicam-se SÓ ao treino (e_treino=True).
        Na validação e no teste o alvo nunca é alterado.
        """
        if not self.ajustado:
            raise RuntimeError("Chama primeiro fit(treino).")

        t = tabela.copy()

        # Outliers: só no treino
        if e_treino and self.tratamento_outliers == "filtrar":
            t = t[~self._mascara_outliers(t)]

        preco = t["price"].copy()
        log_preco = t["log_price"].copy()
        if e_treino and self.tratamento_outliers == "capping":
            log_preco = log_preco.clip(lower=self.limite_baixo, upper=self.limite_alto)
            preco = np.expm1(log_preco)

        # Imputar com o que foi aprendido no treino
        t = self._imputar(t)

        # Construir a matriz de design (coluna a coluna, num dicionário)
        colunas = {}

        # Numéricas: escaladas com a média e o desvio do TREINO
        for col in COLUNAS_NUMERICAS:
            colunas[col] = (t[col] - self.media[col]) / self.desvio[col]

        # Indicadores 0/1: ficam como estão
        for col in COLUNAS_BINARIAS:
            colunas[col] = t[col]

        # Categóricas: one-hot com as MESMAS colunas em todos os conjuntos
        for col in COLUNAS_CATEGORICAS:
            serie = self._agrupar(t[col], col)
            for categoria in self.categorias_dummy[col]:
                colunas[col + "=" + str(categoria)] = (serie == categoria).astype(int)

        # Extras (Etapa 9): padronizados com a média e o desvio do TREINO
        if self.extras:
            brutos = self._extras_brutos(self._numericas_escaladas(t), t)
            for nome, serie in brutos.items():
                colunas[nome] = (serie - self.media_extras[nome]) / self.desvio_extras[nome]

        X = pd.DataFrame(colunas, index=t.index)
        return X, preco, log_preco


# ----------------------------------------------------------------------
# 5) Guardar a matriz de design
# ----------------------------------------------------------------------
def guardar_matriz(X, preco, log_preco, caminho):
    """Guarda X e os alvos num único ficheiro comprimido (.csv.gz)."""
    saida = X.copy()
    saida["price"] = preco
    saida["log_price"] = log_preco
    saida.to_csv(caminho, compression="gzip", index_label="id_linha")


# ----------------------------------------------------------------------
# 5b) Alvo binário "caro" (Etapa 10)
# ----------------------------------------------------------------------
class AlvoCaro:
    """
    Define "caro" = preço acima do percentil `quantil` (80%) dos anúncios COMPARÁVEIS.
    Comparável = mesmo room_type e mesmo concelho (grupo="concelho"), ou mesmo room_type
    e mesmo bairro (grupo="bairro").

    Os limiares aprendem-se SÓ com o treino (como a mediana ou a média: senão há leakage).
    Se um grupo tem menos de `min_grupo` anúncios no treino, usamos o limiar do room_type;
    se nem esse existir, o limiar global. Usa-se sempre o preço ORIGINAL (sem capping).
    """

    def __init__(self, grupo="concelho", quantil=0.80, min_grupo=MIN_CATEGORIA):
        if grupo not in ("concelho", "bairro"):
            raise ValueError("grupo tem de ser 'concelho' ou 'bairro'")
        self.grupo = grupo
        self.quantil = quantil
        self.min_grupo = min_grupo
        self.coluna_geo = "concelho" if grupo == "concelho" else "neighbourhood_cleansed"

    def fit(self, treino):
        preco = treino["price"]
        self.limiar_global = float(preco.quantile(self.quantil))
        self.limiar_tipo = preco.groupby(treino["room_type"]).quantile(self.quantil).to_dict()
        tabela = preco.groupby([treino["room_type"], treino[self.coluna_geo]])
        tamanhos = tabela.size()
        limiares = tabela.quantile(self.quantil)
        self.limiar_grupo = {k: float(v) for k, v in limiares.items() if tamanhos[k] >= self.min_grupo}
        return self

    def limiares(self, tabela):
        """Limiar de cada linha (série com o mesmo índice da tabela)."""
        valores = []
        for tipo, geo in zip(tabela["room_type"], tabela[self.coluna_geo]):
            if (tipo, geo) in self.limiar_grupo:
                valores.append(self.limiar_grupo[(tipo, geo)])
            else:
                valores.append(self.limiar_tipo.get(tipo, self.limiar_global))
        return pd.Series(valores, index=tabela.index, name="limiar_caro")

    def transformar(self, tabela):
        """0/1: 1 se o preço original for superior ao limiar do grupo."""
        return (tabela["price"] > self.limiares(tabela)).astype(int).rename("caro")


def preparar_fold(treino, validacao, estrategia_imputacao="simples", tratamento_outliers="nada",
                  extras=None, grupo_caro="concelho", quantil_caro=0.80):
    """
    Prepara UM fold da validação cruzada: tudo o que aprende (medianas, médias, limites de
    outliers, limiares do "caro") aprende-se SÓ com `treino` e aplica-se a `validacao`.
    Devolve um dicionário com o mesmo formato (nomes) do preparar_tudo.
    """
    pre = PreProcessador(estrategia_imputacao, tratamento_outliers, extras).fit(treino)
    X_treino, preco_treino, log_treino = pre.transformar(treino, e_treino=True)
    X_val, preco_val, log_val = pre.transformar(validacao)
    alvo = AlvoCaro(grupo_caro, quantil_caro).fit(treino)
    return {
        "pre": pre, "alvo_caro": alvo,
        "X_treino": X_treino, "preco_treino": preco_treino, "log_treino": log_treino,
        "y_treino_caro": alvo.transformar(treino).loc[X_treino.index],
        "X_val": X_val, "preco_val": preco_val, "log_val": log_val,
        "y_val_caro": alvo.transformar(validacao).loc[X_val.index],
    }


def preparar_tudo(estrategia_imputacao="simples", tratamento_outliers="nada",
                  guardar=True, caminho_dados=None, pasta_saida=None, extras=None,
                  grupo_caro="concelho", quantil_caro=0.80):
    """Corre o pipeline completo e devolve tudo o que as próximas etapas precisam."""
    df = carregar_dados(caminho_dados)
    feat = criar_features(df)
    treino, validacao, teste, dev = dividir_dados(feat)

    pre = PreProcessador(estrategia_imputacao, tratamento_outliers, extras)
    pre.fit(treino)

    X_treino, preco_treino, log_treino = pre.transformar(treino, e_treino=True)
    X_val, preco_val, log_val = pre.transformar(validacao)
    X_teste, preco_teste, log_teste = pre.transformar(teste)

    # Alvo "caro": limiares só do treino; alinhado com as linhas de X (importa no "filtrar")
    alvo = AlvoCaro(grupo_caro, quantil_caro).fit(treino)
    y_treino_caro = alvo.transformar(treino).loc[X_treino.index]
    y_val_caro = alvo.transformar(validacao).loc[X_val.index]
    y_teste_caro = alvo.transformar(teste).loc[X_teste.index]

    if guardar:
        pasta = Path(pasta_saida) if pasta_saida else PASTA_PROCESSADOS
        pasta.mkdir(parents=True, exist_ok=True)
        etiqueta = estrategia_imputacao + "_" + tratamento_outliers
        if extras:
            etiqueta = etiqueta + "_" + "+".join(extras)
        guardar_matriz(X_treino, preco_treino, log_treino, pasta / ("matriz_" + etiqueta + "_treino.csv.gz"))
        guardar_matriz(X_val, preco_val, log_val, pasta / ("matriz_" + etiqueta + "_validacao.csv.gz"))
        guardar_matriz(X_teste, preco_teste, log_teste, pasta / ("matriz_" + etiqueta + "_teste.csv.gz"))

    return {
        "pre": pre, "alvo_caro": alvo,
        "y_treino_caro": y_treino_caro, "y_val_caro": y_val_caro, "y_teste_caro": y_teste_caro,
        "treino": treino, "validacao": validacao, "teste": teste, "dev": dev,
        "X_treino": X_treino, "preco_treino": preco_treino, "log_treino": log_treino,
        "X_val": X_val, "preco_val": preco_val, "log_val": log_val,
        "X_teste": X_teste, "preco_teste": preco_teste, "log_teste": log_teste,
    }


if __name__ == "__main__":
    resultado = preparar_tudo("simples", "nada")
    print("Treino:", resultado["X_treino"].shape)
    print("Validação:", resultado["X_val"].shape)
    print("Teste:", resultado["X_teste"].shape)
    print("Valores em falta no treino:", int(resultado["X_treino"].isnull().sum().sum()))