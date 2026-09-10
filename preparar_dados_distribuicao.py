# -*- coding: utf-8 -*-
"""
preparar_dados_distribuicao.py
--------------------------------------------------------------------------
Módulo 6 (novo) — Distribuição e Logística do PNAE — Salvador/BA

Cruza três saídas do pipeline existente:
  - cardapio_mensal_todos_grupos.csv  (Módulo 5, per capita)
  - demanda_nutricional_escolas.csv   (Módulo 2, por escola)
  - resumo_precos_referencia.csv      (Módulo 3, origem do preço)

e produz três bases agregadas, uma para cada documento solicitado:
  1. municipal_consolidado.csv   -> Doc 1 (Secretaria)
  2. distribuicao_por_escola.csv -> Doc 2 (cardápio + quantidade por escola)
  3. logistica_entregas.csv      -> Doc 3 (entregas por escola/mês/semana)

⚠️ IMPORTANTE (documentar no TCC): cardapio_mensal_todos_grupos.csv usado
aqui é uma AMOSTRA de 2 meses (mar/set) por custo computacional do solver.
Os totais aqui gerados cobrem SOMENTE esses 2 meses — não são anuais.
Quando o solver rodar para os 10 meses (fev até dez - jun), basta apontar
CARDAPIO_PATH para o CSV completo e reexecutar este script; toda a lógica
de junção e agregação permanece válida.

Convenção de match alimento -> preço de referência: normalização de texto
(remover acentos, minúsculas, remover termos de preparo "cru/crua/cozido")
+ correspondência aproximada (rapidfuzz). Todo match abaixo de
SCORE_MINIMO_MATCH é sinalizado como "NÃO VERIFICADO" em vez de aceito
silenciosamente — mesmo padrão de transparência já usado no projeto
(campo base_af / confianca_match do Módulo 3).
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).parent
CARDAPIO_PATH = BASE_DIR / "data/output/cardapio_mensal_todos_grupos.csv"
DEMANDA_PATH = BASE_DIR / "data/output/demanda_nutricional_escolas.csv"
PRECOS_PATH = BASE_DIR / "data/output/resumo_precos_referencia.csv"

OUTPUT_DIR = BASE_DIR / "data/output"
OUTPUT_DIR.mkdir(exist_ok=True)

SCORE_MINIMO_MATCH = 85  # abaixo disso -> "NÃO VERIFICADO"

MESES_AMOSTRADOS = None  # preenchido após carregar o cardápio


# ═══════════════════════════════════════════════════════════════════════
# NORMALIZAÇÃO DE TEXTO (para casar alimento <-> produto_busca)
# ═══════════════════════════════════════════════════════════════════════

_PREPARO_RE = re.compile(
    r"\b(cru|crua|crus|cruas|cozid[oa]s?|assad[oa]s?|frit[oa]s?)\b"
)


def normaliza(txt: str) -> str:
    txt = unicodedata.normalize("NFD", str(txt))
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    txt = txt.lower()
    txt = re.sub(r"[,.]", " ", txt)
    txt = _PREPARO_RE.sub("", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def construir_mapa_origem_precos(alimentos: list[str], df_precos: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada 'alimento' (nome TACO do cardápio), encontra o produto de
    referência de preço mais próximo em resumo_precos_referencia.csv e
    traz fonte_ref / base_af / confianca_match / score de correspondência.
    """
    candidatos = df_precos["produto_busca"].tolist()
    linhas = []
    for alimento in alimentos:
        alimento_norm = normaliza(alimento)
        resultado = process.extractOne(alimento_norm, candidatos, scorer=fuzz.WRatio)
        if resultado is None:
            linhas.append(
                dict(
                    alimento=alimento,
                    produto_busca_match=None,
                    score_match=0,
                    status_match="NÃO ENCONTRADO",
                    fonte_ref=None,
                    base_af=None,
                    confianca_match_origem=None,
                )
            )
            continue
        produto_match, score, _ = resultado
        linha_preco = df_precos.loc[df_precos["produto_busca"] == produto_match].iloc[0]
        status = "OK" if score >= SCORE_MINIMO_MATCH else "NÃO VERIFICADO"
        linhas.append(
            dict(
                alimento=alimento,
                produto_busca_match=produto_match,
                score_match=round(score, 1),
                status_match=status,
                fonte_ref=linha_preco["fonte_ref"],
                base_af=linha_preco["base_af"],
                confianca_match_origem=linha_preco["confianca_match"],
            )
        )
    return pd.DataFrame(linhas)


# ═══════════════════════════════════════════════════════════════════════
# CARGA
# ═══════════════════════════════════════════════════════════════════════

def carregar_bases():
    df_cardapio = pd.read_csv(CARDAPIO_PATH)

    df_demanda = pd.read_csv(
        DEMANDA_PATH, sep=";", decimal=",", encoding="utf-8-sig"
    )
    df_demanda.columns = [c.strip() for c in df_demanda.columns]

    df_precos = pd.read_csv(PRECOS_PATH)

    return df_cardapio, df_demanda, df_precos


# ═══════════════════════════════════════════════════════════════════════
# JUNÇÕES
# ═══════════════════════════════════════════════════════════════════════

def preparar(df_cardapio, df_demanda, df_precos):
    # --- 1. chave de grupo compatível: grupo_nutricional + num_refeicoes -> "GRUPO_NREF"
    df_demanda = df_demanda.copy()
    df_demanda["grupo"] = (
        df_demanda["grupo_nutricional"].astype(str)
        + "_"
        + df_demanda["num_refeicoes"].astype(int).astype(str)
        + "REF"
    )

    # --- 2. mapa de origem de preço por alimento (uma linha por alimento único)
    alimentos_unicos = sorted(df_cardapio["alimento"].unique())
    df_mapa_origem = construir_mapa_origem_precos(alimentos_unicos, df_precos)

    n_nao_verificado = (df_mapa_origem["status_match"] != "OK").sum()
    if n_nao_verificado:
        print(
            f"[AVISO] {n_nao_verificado} alimento(s) com correspondência de preço "
            f"abaixo do score mínimo ({SCORE_MINIMO_MATCH}) — marcados como "
            f"'NÃO VERIFICADO' na coluna status_match. Revisar antes da entrega."
        )

    df_cardapio = df_cardapio.merge(df_mapa_origem, on="alimento", how="left")

    # --- 3. junta cardápio (per capita) com demanda (por escola), pela chave "grupo"
    #     each cardapio row (grupo/mes/dia/alimento) x each school with that grupo
    df_join = df_cardapio.merge(
        df_demanda[
            ["CO_ENTIDADE", "NO_ENTIDADE", "grupo", "grupo_nutricional",
             "num_refeicoes", "quantidade_alunos"]
        ],
        on="grupo",
        how="inner",  # só mantém combinações grupo/mes que existem na amostra do cardápio
    )

    # --- 4. escala per capita -> quantidade real da escola
    df_join["qtd_kg_escola"] = (df_join["qtd_g"] / 1000.0) * df_join["quantidade_alunos"]
    df_join["custo_total_escola"] = df_join["custo_r"] * df_join["quantidade_alunos"]

    return df_join, df_mapa_origem


# ═══════════════════════════════════════════════════════════════════════
# DOC 1 — CONSOLIDADO MUNICIPAL (Secretaria)
# ═══════════════════════════════════════════════════════════════════════

def montar_consolidado_municipal(df_join, df_mapa_origem):
    # 1a. lista de produtos: quantidade total (kg) e custo total por mês, + origem do preço
    produtos_mes = (
        df_join.groupby(["mes", "nome_mes", "alimento", "categoria"], as_index=False)
        .agg(
            quantidade_total_kg=("qtd_kg_escola", "sum"),
            custo_total_r=("custo_total_escola", "sum"),
            escolas_atendidas=("CO_ENTIDADE", "nunique"),
        )
    )
    produtos_mes = produtos_mes.merge(
        df_mapa_origem[["alimento", "fonte_ref", "base_af", "confianca_match_origem", "status_match"]],
        on="alimento",
        how="left",
    )
    produtos_mes = produtos_mes.sort_values(["mes", "custo_total_r"], ascending=[True, False])

    # 1b. resumo financeiro por mês/grupo
    # CORREÇÃO (set/2026): 'quantidade_alunos' se repete em toda linha
    # escola×alimento×dia — somar direto do df_join multiplicava o mesmo
    # aluno várias vezes. O número de alunos precisa vir de uma tabela
    # deduplicada por (mes, grupo, CO_ENTIDADE) antes de somar.
    alunos_por_escola_grupo_mes = (
        df_join[["mes", "nome_mes", "grupo", "CO_ENTIDADE", "quantidade_alunos"]]
        .drop_duplicates()
    )
    resumo_mes_grupo = df_join.groupby(
        ["mes", "nome_mes", "grupo"], as_index=False
    ).agg(custo_total_r=("custo_total_escola", "sum"))

    alunos_agg = alunos_por_escola_grupo_mes.groupby(
        ["mes", "nome_mes", "grupo"], as_index=False
    ).agg(
        alunos_atendidos=("quantidade_alunos", "sum"),
        escolas=("CO_ENTIDADE", "nunique"),
    )
    resumo_mes_grupo = resumo_mes_grupo.merge(
        alunos_agg, on=["mes", "nome_mes", "grupo"], how="left"
    )

    # 1c. % AF real (ponderado por quantidade real, não por linha do cardápio)
    df_join_af = df_join.copy()
    total_custo = df_join_af["custo_total_escola"].sum()
    custo_af = df_join_af.loc[df_join_af["af"] == True, "custo_total_escola"].sum()
    pct_af_real = 100 * custo_af / total_custo if total_custo else 0.0

    custo_total_periodo_amostrado = total_custo

    return produtos_mes, resumo_mes_grupo, pct_af_real, custo_total_periodo_amostrado


# ═══════════════════════════════════════════════════════════════════════
# DOC 2 — DISTRIBUIÇÃO POR ESCOLA (cardápio + quantidade)
# ═══════════════════════════════════════════════════════════════════════

def montar_distribuicao_escola(df_join):
    cols = [
        "CO_ENTIDADE", "NO_ENTIDADE", "grupo", "grupo_nutricional", "num_refeicoes",
        "mes", "nome_mes", "semana", "dia_semana", "dia_mes",
        "alimento", "categoria",
        "quantidade_alunos", "qtd_g", "qtd_kg_escola", "custo_total_escola", "af",
    ]
    df_distribuicao = df_join[cols].sort_values(
        ["NO_ENTIDADE", "grupo", "mes", "dia_mes", "alimento"]
    )
    return df_distribuicao


# ═══════════════════════════════════════════════════════════════════════
# DOC 3 — LOGÍSTICA DE ENTREGAS (por escola / mês / semana)
# ═══════════════════════════════════════════════════════════════════════

def montar_logistica_entregas(df_join):
    df_logistica = (
        df_join.groupby(
            ["CO_ENTIDADE", "NO_ENTIDADE", "mes", "nome_mes", "semana", "alimento", "categoria"],
            as_index=False,
        )
        .agg(
            quantidade_total_kg=("qtd_kg_escola", "sum"),
            custo_total_r=("custo_total_escola", "sum"),
            dias_de_entrega_na_semana=("dia_mes", "nunique"),
        )
        .sort_values(["NO_ENTIDADE", "mes", "semana", "alimento"])
    )
    return df_logistica


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    df_cardapio, df_demanda, df_precos = carregar_bases()

    global MESES_AMOSTRADOS
    MESES_AMOSTRADOS = sorted(df_cardapio["mes"].unique().tolist())

    df_join, df_mapa_origem = preparar(df_cardapio, df_demanda, df_precos)

    produtos_mes, resumo_mes_grupo, pct_af_real, custo_total_amostrado = (
        montar_consolidado_municipal(df_join, df_mapa_origem)
    )
    df_distribuicao = montar_distribuicao_escola(df_join)
    df_logistica = montar_logistica_entregas(df_join)

    # exporta bases intermediárias (para auditoria e para os geradores de doc)
    produtos_mes.to_csv(OUTPUT_DIR / "doc1_produtos_por_mes.csv", index=False)
    resumo_mes_grupo.to_csv(OUTPUT_DIR / "doc1_resumo_mes_grupo.csv", index=False)
    df_distribuicao.to_csv(OUTPUT_DIR / "doc2_distribuicao_por_escola.csv", index=False)
    df_logistica.to_csv(OUTPUT_DIR / "doc3_logistica_entregas.csv", index=False)
    df_mapa_origem.to_csv(OUTPUT_DIR / "auditoria_mapa_origem_precos.csv", index=False)

    print("\n=== RESUMO DA PREPARAÇÃO ===")
    print(f"Meses amostrados no cardápio: {MESES_AMOSTRADOS}")
    print(f"Escolas cobertas (grupos presentes na amostra): {df_join['CO_ENTIDADE'].nunique()}")
    print(f"Linhas — distribuição por escola: {len(df_distribuicao):,}")
    print(f"Linhas — logística de entregas: {len(df_logistica):,}")
    print(f"Custo total REAL do cardápio — período amostrado (mar/set): R$ {custo_total_amostrado:,.2f}")
    print(f"% AF real (ponderado por quantidade real): {pct_af_real:.1f}%")
    print(f"Alimentos com match de preço não verificado: "
          f"{(df_mapa_origem['status_match'] != 'OK').sum()} de {len(df_mapa_origem)}")

    return dict(
        df_join=df_join,
        produtos_mes=produtos_mes,
        resumo_mes_grupo=resumo_mes_grupo,
        pct_af_real=pct_af_real,
        custo_total_amostrado=custo_total_amostrado,
        df_distribuicao=df_distribuicao,
        df_logistica=df_logistica,
        df_mapa_origem=df_mapa_origem,
        meses_amostrados=MESES_AMOSTRADOS,
    )


if __name__ == "__main__":
    main()
