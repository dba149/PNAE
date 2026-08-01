"""
otimizacao_pnae_solver.py
═══════════════════════════════════════════════════════════════════════════════
Motor de otimização do PNAE — Programação Linear Inteira Mista (MILP)
Horizonte: 1 mês letivo (20 dias úteis = 4 semanas × 5 dias)

Projeto: Otimização inteligente do PNAE
Aluna  : Daniela Barreto Araújo
Orient.: Renato Cividini Matthiesen
Fonte  : Resolução CD/FNDE nº 4, de 26 de fevereiro de 2026
═══════════════════════════════════════════════════════════════════════════════

NOTA METODOLÓGICA — Meses processados
Janeiro é excluído por ser mês de férias escolares no calendário letivo
municipal de Salvador-BA. Os meses processados são: fev, mar, abr, mai,
jun, jul, ago, set, out, nov, dez (11 meses).

NOTA METODOLÓGICA — Restrições mensais via horizonte de 20 dias
As restrições da Resolução que possuem limite mensal (produtos cárneos ≤ 2×,
doces ≤ 1×, conservas ≤ 1× etc.) são modeladas diretamente no horizonte de
20 dias úteis, sem necessidade de conversão para proxy semanal.

NOTA METODOLÓGICA — Variáveis binárias de presença
A introdução de y_{jd} ∈ {0,1} transforma o modelo de LP para MILP.
O solver CBC (COIN-OR) resolve MILP nativamente. Para |J|~500 alimentos e
D=20 dias, cada instância tem ~10.000 variáveis binárias + ~10.000 contínuas.
Tempo esperado por instância: 10–60 s no CBC em modo padrão.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from pulp import (
    LpProblem, LpMinimize, LpVariable, LpStatus,
    lpSum, value, PULP_CBC_CMD,
)

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pnae_solver")

# ── Constantes globais ────────────────────────────────────────────────────────
MESES_LETIVOS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # jan=1 excluído (férias)
NOMES_MESES = {
    2:"Fevereiro", 3:"Março", 4:"Abril", 5:"Maio", 6:"Junho",
    7:"Julho",    8:"Agosto", 9:"Setembro", 10:"Outubro",
    11:"Novembro", 12:"Dezembro",
}
DIAS_UTEIS_MES   = 20   # 4 semanas × 5 dias letivos
SEMANAS_MES      = 4
DIAS_SEMANA      = 5
M_BIG            = 600.0  # Big-M para vinculação x ↔ y (g máximo por item/dia)
EPS_MIN_G        = 10.0    # Quantidade mínima se o alimento for selecionado (g)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARREGAMENTO DE DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def carregar_bases(output_folder: str = "./data/output/") -> dict:
    """
    Carrega todos os CSVs e JSONs dos módulos anteriores.
    Retorna dicionário com os DataFrames e o objeto de restrições.
    """
    def _csv(nome, sep=",", dec="."):
        path = os.path.join(output_folder, nome)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}\n"
                f"Execute o módulo de origem antes de rodar o solver."
            )
        df = pd.read_csv(path, sep=sep, decimal=dec, low_memory=False)
        log.info("%-52s → %d linhas × %d colunas", nome, *df.shape)
        return df

    log.info("Carregando bases dos módulos anteriores...")
    dados = {
        "alimentos" : _csv("alimentos_elegiveis_otimizador.csv"),
        "precos"    : _csv("resumo_precos_referencia.csv"),
        "params"    : _csv("parametros_nutricionais_solver.csv"),
        "demanda"   : _csv("demanda_nutricional_escolas.csv", sep=";", dec=","),
        "sazonalidade": _csv("sazonalidade_por_produto.csv")
            if os.path.exists(os.path.join(output_folder, "sazonalidade_por_produto.csv"))
            else pd.DataFrame(),
    }
    json_path = os.path.join(output_folder, "restricoes_pnae.json")
    with open(json_path, encoding="utf-8") as f:
        dados["restricoes"] = json.load(f)
    log.info("%-52s → %d seções", "restricoes_pnae.json", len(dados["restricoes"]))
    return dados


def preparar_dataset_solver(dados: dict) -> pd.DataFrame:
    """
    Constrói o dataset consolidado de alimentos para o solver:
    - Exclui proibidos e ingredientes de receita (is_ingrediente=True)
    - Faz join com preços (preco_kg_referencia do Módulo 3)
    - Normaliza flag AF
    - Converte preço para R$/g
    - Resolve colunas nutricionais por alias
    """
    df = dados["alimentos"].copy()
    df_precos = dados["precos"].copy()

    # ── Excluir proibidos ─────────────────────────────────────────────────────
    if "restricao_pnae" in df.columns:
        df = df[df["restricao_pnae"] != "Aquisição proibida"].copy()

    # ── Excluir ingredientes de receita ───────────────────────────────────────
    if "is_ingrediente" in df.columns:
        n_ing = df["is_ingrediente"].sum()
        df = df[~df["is_ingrediente"].astype(bool)].copy()
        log.info("Ingredientes de receita excluídos do solver: %d", n_ing)
    else:
        log.warning(
            "COLUNA 'is_ingrediente' AUSENTE em alimentos_elegiveis_otimizador.csv.\n"
            "  Isso significa que temperos, gorduras e farinhas entrarão no solver\n"
            "  como alimentos autônomos, gerando cardápios inválidos.\n"
            "  → Corrija executando em sequência:\n"
            "    1. otimizacao_pnae_alimentos.ipynb  (gera is_ingrediente)\n"
            "    2. otimizacao_pnae_regras.ipynb         (Bloco 8: adicionar\n"
            "       'is_ingrediente' e 'motivo_ingrediente' em cols_extra e\n"
            "       'produto_busca' em cols_controle)\n"
            "  O solver continuará, mas os resultados podem conter ingredientes isolados."
        )

    # ── Join com preços ───────────────────────────────────────────────────────
    # Chave canônica: 'produto_busca' (Módulo 1 cria produto_busca = produto_base;
    # Módulo 3 exporta resumo_precos_referencia.csv com produto_busca como índice;
    # Módulo 4 deve preservar produto_busca em cols_controle ao exportar
    # alimentos_elegiveis_otimizador.csv — verificar se está incluída lá).
    # CORREÇÃO (jul/2026): o Módulo 4 agora preserva 'produto_busca' explicitamente
    # em alimentos_elegiveis_otimizador.csv (ver otimizacao_pnae_regras.ipynb, Bloco 4).
    # Mantemos 'produto_base' como fallback defensivo caso uma versão anterior do
    # Módulo 4 (sem essa correção) ainda esteja em uso.
    col_join = next(
        (c for c in ["produto_busca", "produto_base"] if c in df.columns and c in df_precos.columns),
        None,
    )
    if col_join:
        cols_trazer = [c for c in ["preco_kg_referencia","is_af","confianca_af"]
                       if c in df_precos.columns]
        df = df.merge(
            df_precos[[col_join] + cols_trazer].drop_duplicates(col_join),
            on=col_join, how="left", suffixes=("", "_conab"),
        )
        n_com_preco = df["preco_kg_referencia"].notna().sum() \
                      if "preco_kg_referencia" in df.columns else 0
        log.info("Join preços por '%s': %d/%d alimentos com preço", col_join, n_com_preco, len(df))
        if n_com_preco == 0:
            log.warning(
                "JOIN REALIZADO MAS ZERO ALIMENTOS RECEBERAM PREÇO.\n"
                "  Causa mais provável: coluna 'produto_busca' existe em ambos os\n"
                "  dataframes, mas os valores não coincidem (normalização diferente).\n"
                "  Amostra alimentos_elegiveis: %s\n"
                "  Amostra resumo_precos      : %s\n"
                "  → Verifique se o Módulo 4 preserva 'produto_busca' ao exportar\n"
                "    alimentos_elegiveis_otimizador.csv (deve estar em cols_controle).",
                df[col_join].dropna().head(3).tolist(),
                df_precos[col_join].dropna().head(3).tolist(),
            )
    else:
        # produto_busca ausente em um dos dataframes
        cols_alim  = [c for c in df.columns if "produto" in c.lower() or "alimento" in c.lower()]
        cols_preco = [c for c in df_precos.columns if "produto" in c.lower() or "alimento" in c.lower()]
        log.warning(
            "COLUNA 'produto_busca' NÃO ENCONTRADA PARA JOIN DE PREÇOS.\n"
            "  Colunas candidatas em alimentos_elegiveis : %s\n"
            "  Colunas candidatas em resumo_precos       : %s\n"
            "  → Verifique se o Módulo 4 inclui 'produto_busca' em cols_controle\n"
            "    no Bloco 8 (Dataset consolidado para o otimizador).",
            cols_alim, cols_preco,
        )

    # ── Flag AF ───────────────────────────────────────────────────────────────
    if "is_af" in df.columns:
        df["agricultura_familiar"] = df["is_af"].apply(
            lambda v: v is True or str(v).lower() == "true"
        )
    elif "agricultura_familiar" not in df.columns:
        df["agricultura_familiar"] = False

    # CORREÇÃO (jul/2026): diagnóstico explícito de cobertura — antes, uma
    # cobertura de AF/preço anormalmente baixa (ex.: causada por um CSV
    # desatualizado do Módulo 4, produzido antes da correção do join) passava
    # despercebida, pois o solver seguia rodando normalmente com AF=False e
    # preço imputado para praticamente todos os itens. Tornamos isso visível.
    n_af = int(df["agricultura_familiar"].sum())
    pct_af = n_af / len(df) * 100 if len(df) else 0
    log.info("Alimentos classificados como Agricultura Familiar: %d/%d (%.1f%%)", n_af, len(df), pct_af)
    if pct_af < 5:
        log.warning(
            "COBERTURA DE AGRICULTURA FAMILIAR SUSPEITOSAMENTE BAIXA (%.1f%%).\n"
            "  Causas prováveis: (1) 'alimentos_elegiveis_otimizador.csv' foi gerado\n"
            "  ANTES da correção de join do Módulo 4 (reexecute os Módulos 3→4→5\n"
            "  em sequência); (2) baixa cobertura de match CONAB/planilha manual\n"
            "  para os produtos do cardápio — verifique 'produtos_sem_preco.csv'.",
            pct_af,
        )

    # ── Preço R$/g ────────────────────────────────────────────────────────────
    col_kg = next(
        (c for c in ["preco_kg_referencia","preco_kg_ref","preco_referencia"]
         if c in df.columns), None
    )
    if col_kg is None:
        raise ValueError("Coluna de preço não encontrada. Verifique o join com Módulo 3.")
    n_com_preco_bruto = int(pd.to_numeric(df[col_kg], errors="coerce").notna().sum())
    pct_preco_bruto = n_com_preco_bruto / len(df) * 100 if len(df) else 0
    log.info("Alimentos com preço de referência: %d/%d (%.1f%%)",
              n_com_preco_bruto, len(df), pct_preco_bruto)
    if pct_preco_bruto < 20:
        log.warning(
            "COBERTURA DE PREÇO DE REFERÊNCIA SUSPEITOSAMENTE BAIXA (%.1f%%).\n"
            "  Reexecute os Módulos 3→4→5 em sequência e/ou complemente\n"
            "  'produtos_sem_preco.csv' na planilha manual (Módulo 3, Bloco 12B).",
            pct_preco_bruto,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # AJUSTE (jul/2026) — impacto da mudança no Módulo 3 (Bloco 12C /
    # produtos_aptos_solver.csv): alimentos sem NENHUM preço de referência
    # (nem CONAB, nem planilha manual) deixam de ser IMPUTADOS com a mediana
    # de preço da categoria e passam a ser REMOVIDOS da base do solver.
    # ─────────────────────────────────────────────────────────────────────
    # Causa-raiz do comportamento anterior: um alimento sem preço real virava
    # NaN no join (linha 155) e, em seguida, era silenciosamente atribuído a
    # um preço artificial (mediana da categoria) — ou seja, um item sem custo
    # de mercado conhecido podia ser escolhido pelo solver como se tivesse um
    # custo real. Isso é inconsistente com o critério "correção > cobertura"
    # já adotado no Módulo 3 (itens sem match confiável vão para
    # produtos_sem_preco.csv, não entram no cálculo).
    # Com o Módulo 4 agora removendo esses itens antes de exportar
    # alimentos_elegiveis_otimizador.csv, o filtro abaixo normalmente não
    # remove nada — é mantido aqui como salvaguarda defensiva para o caso de
    # o solver ser executado com uma versão desatualizada desse CSV.
    df["preco_g"] = pd.to_numeric(df[col_kg], errors="coerce") / 1000
    sem_preco_real = df["preco_g"].isna() | (df["preco_g"] <= 0)
    n_removidos_sem_preco = int(sem_preco_real.sum())
    if n_removidos_sem_preco:
        log.warning(
            "%d alimento(s) sem preço de referência real foram REMOVIDOS da base "
            "do solver (não são mais imputados por mediana de categoria). "
            "Isso não deveria ocorrer se o Módulo 4 (versão atual) já os "
            "removeu — reexecute os Módulos 3→4→5 em sequência para confirmar.",
            n_removidos_sem_preco,
        )
        df = df[~sem_preco_real].copy()

    # ── Resolver aliases de colunas nutricionais ──────────────────────────────
    ALIASES = {
        "energia"     : ["energia_kcal","energia","kcal"],
        "proteina"    : ["proteina_g","proteina","prot"],
        "carboidratos": ["carboidratos_g","carboidratos","carb"],
        "lipideos"    : ["lipideos_g","lipideos","gordura","lip"],
        "sodio"       : ["sodio_mg","sodio","na"],
        "ferro"       : ["ferro_mg","ferro","fe"],
        "vitamina_a"  : ["vitamina_a_mcg","vitamina_a","retinol","vit_a"],
        "vitamina_c"  : ["vitamina_c_mg","vitamina_c","vit_c"],
        "calcio"      : ["calcio_mg","calcio","ca"],
    }
    col_map = {}
    for nut, candidates in ALIASES.items():
        found = next((c for c in candidates if c in df.columns), None)
        if found:
            col_map[nut] = found
            df[nut] = pd.to_numeric(df[found], errors="coerce").fillna(0)
        else:
            df[nut] = 0.0
            log.warning("Nutriente '%s' não encontrado — coluna zerada.", nut)

    df = df.reset_index(drop=True)
    df.index.name = "idx"
    log.info("Dataset solver pronto: %d alimentos elegíveis", len(df))
    return df, col_map


def aplicar_sazonalidade(df: pd.DataFrame, df_saz: pd.DataFrame, mes: int) -> pd.DataFrame:
    """
    Ajusta o preço de referência pelo índice sazonal do mês para cada produto.
    Se a sazonalidade não estiver disponível, retorna df sem alteração.
    """
    if df_saz.empty or mes not in range(1, 13):
        return df

    df_out = df.copy()
    col_join = next((c for c in ["produto_busca","alimento"] if c in df.columns
                     and c in df_saz.columns), None)
    if col_join is None:
        return df_out

    col_indice = "indice_sazonal_por_mes"
    if col_indice not in df_saz.columns:
        return df_out

    # Extrair índice do mês (stored como dict ou string)
    def _get_indice(val):
        try:
            d = val if isinstance(val, dict) else json.loads(val.replace("'", '"'))
            return float(d.get(mes, d.get(str(mes), 1.0)))
        except Exception:
            return 1.0

    df_saz_mes = df_saz[[col_join, col_indice]].copy()
    df_saz_mes["fator_sazonal"] = df_saz_mes[col_indice].apply(_get_indice)
    df_saz_mes = df_saz_mes[[col_join, "fator_sazonal"]].drop_duplicates(col_join)

    df_out = df_out.merge(df_saz_mes, on=col_join, how="left")
    df_out["fator_sazonal"] = df_out["fator_sazonal"].fillna(1.0)
    df_out["preco_g"] = df_out["preco_g"] * df_out["fator_sazonal"]

    log.info("Sazonalidade aplicada para mês %02d (%s)", mes, NOMES_MESES.get(mes, ""))
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARÂMETROS DO GRUPO NUTRICIONAL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerfilNutricional:
    """
    Encapsula todos os parâmetros de restrição para um grupo nutricional.
    Carregado a partir de parametros_nutricionais_solver.csv e restricoes_pnae.json.
    """
    grupo          : str
    num_refeicoes  : int
    periodo        : str          # 'parcial' ou 'integral'
    pct_cobertura  : float
    energia_ref    : float        # kcal/dia de referência
    sodio_max_mg   : float
    diversidade_min: int
    flv_min_g_semana: float
    frutas_dias_min : int
    legumes_dias_min: int
    ferro_dias_min  : int
    vit_a_dias_min  : int
    # Limites nutricionais diários (já multiplicados por pct_cobertura)
    min_energia    : float = 0.0
    min_proteina   : float = 0.0
    min_carboidratos: float = 0.0
    min_ferro      : float = 0.0
    min_vitamina_a : float = 0.0
    min_vitamina_c : float = 0.0
    min_calcio     : Optional[float] = None
    # Macronutrientes absolutos (derivados do VET)
    lip_min_g      : float = 0.0
    lip_max_g      : float = 0.0
    # Limites mensais de frequência (em 20 dias úteis)
    max_carneos_mes     : int = 2
    max_conservas_mes   : int = 1
    max_doces_mes       : int = 1
    max_lacteos_adic_mes: int = 1
    max_biscoito_pao_semana: int = 2
    max_prep_doces_mes  : int = 2
    max_margarina_mes   : int = 2
    # Agricultura Familiar
    pct_af_meta    : float = 0.45
    # CORREÇÃO (jul/2026): Lei nº 15.226/2025 elevou o mínimo legal de
    # Agricultura Familiar de 30% para 45%, vigente desde jan/2026.
    pct_af_legal   : float = 0.45


def montar_perfil(grupo: str, df_params: pd.DataFrame, restricoes: dict) -> PerfilNutricional:
    """
    Constrói um PerfilNutricional a partir do grupo nutricional solicitado.
    """
    row = df_params[df_params["grupo_nutricional"] == grupo]
    if row.empty:
        disponiveis = df_params["grupo_nutricional"].unique().tolist()
        raise ValueError(f"Grupo '{grupo}' não encontrado. Disponíveis: {disponiveis}")
    row = row.iloc[0]

    pct   = float(row.get("pct_nutricional", 0.20))
    sodio = float(row.get("sodio_max_mg", 600))

    # ═════════════════════════════════════════════════════════════════════
    # CORREÇÃO (jul/2026): período e nº de refeições eram adivinhados por
    # substring no NOME do grupo (ex.: "integral" in grupo, "2" in grupo),
    # totalmente desconectado do valor real de num_refeicoes vindo do CSV.
    # Como nomes de grupo no dataset atual são 'CRECHE', 'FUND_11_15',
    # 'PRE_ESCOLA' etc. (sem a palavra "integral" nem dígito "2"), TODO grupo
    # caía no fallback n_ref=1 / periodo="parcial" — mesmo quando a linha em
    # df_params tinha num_refeicoes=3 (ex.: CRECHE com 3 refeições era
    # tratada como periodo="parcial", 1 refeição, para fins de FLV semanal,
    # frequência de frutas/legumes e limites mensais de biscoito/lácteos/
    # preparações doces/margarina — todos calculados com o valor errado).
    #
    # A diversidade mínima (`diversidade_min`) NÃO era afetada por esse bug
    # específico, pois seu cálculo já usava `pct` (correto), e não este
    # `n_ref` local — por isso pode ter parecido "23" corretamente ao lado
    # de um "Refeições/dia" exibido errado (1) na mesma tabela-resumo.
    #
    # Correção: usar num_refeicoes diretamente da linha de df_params (já
    # correto, propagado por otimizacao_pnae_regras.ipynb) para determinar
    # tanto n_ref quanto periodo.
    # ═════════════════════════════════════════════════════════════════════
    n_ref   = int(row.get("num_refeicoes", 1))
    periodo = "integral" if n_ref >= 3 else "parcial"

    # FLV semanal
    flv_g = restricoes["flv_semana_gramas"][periodo]

    # Frequências mínimas de grupos por semana (5 dias)
    freq = restricoes["flv_frequencia_minima_dias"]
    freq_micro = restricoes["micronutrientes_frequencia_minima_dias"]

    # Diversidade mínima
    div_map = restricoes["diversidade_minima_alimentos"]
    if pct >= 0.70:
        div_min = div_map["cardapio_3_refeicoes_ou_70pct"]
    elif pct >= 0.30:
        div_min = div_map["cardapio_2_refeicoes_ou_30pct"]
    else:
        div_min = div_map["cardapio_1_refeicao_ou_20pct"]

    def _lim_min(col):
        c = f"min_{col}"
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
        return None

    def _lim_max(col):
        c = f"max_{col}"
        if c in row.index and pd.notna(row[c]):
            return float(row[c])
        return None

    # ═════════════════════════════════════════════════════════════════════
    # CORREÇÃO (jul/2026): energia e lipídios agora vêm diretamente do valor
    # já calculado por grupo_nutricional × num_refeicoes em
    # parametros_nutricionais_solver.csv (originário de demanda_nutricional_
    # escolas.csv / Módulo Censo), que já reflete a tabela de referência FNDE
    # (ex.: Creche/3 refeições → energia 708 kcal, lipídios 20–28 g).
    #
    # Causa-raiz do bug anterior: `en_ref = en_min / pct` reinflava a energia
    # de referência (708 / 0.70 = 1011,4 kcal) como se 708 fosse só uma fração
    # do VET diário total, quando na verdade 708 já É o valor de referência
    # do grupo. Isso, combinado com o uso do %VET genérico (15–30 %) de
    # `limites_macronutrientes_pct_vet` — que não corresponde ao intervalo
    # real por grupo/refeições (25–35 % na tabela FNDE) — produzia limites de
    # lipídio incorretos e descartava os valores `lipidio_min`/`lipidio_max`
    # já calculados e disponíveis no CSV.
    #
    # Correção: usar energia e lipídios diretamente das colunas min_/max_ já
    # propagadas por otimizacao_pnae_regras.ipynb (Bloco 5), sem qualquer
    # reponderação. `limites_macronutrientes_pct_vet` deixa de ser usado
    # aqui (a exigência específica por grupo já está coberta pelos valores
    # do CSV); o dicionário permanece em REGRAS_PNAE apenas como referência
    # documental da regra geral do art. 19 da Resolução.
    # ═════════════════════════════════════════════════════════════════════
    en_ref  = _lim_min("energia_kcal") or pct * 550
    lip_min = _lim_min("lipideos_g") or (0.15 * en_ref) / 9
    lip_max = _lim_max("lipideos_g") or (0.30 * en_ref) / 9

    # Limites mensais (conforme período — usar os mais restritivos)
    lm = restricoes["limites_maximos"]
    lac_mes = (lm["lacteos_aditivados_integral_vezes_mes"]
               if periodo == "integral" else lm["lacteos_aditivados_parcial_vezes_mes"])
    bis_sem = (lm["biscoito_pao_bolo_integral_3ref_vezes_semana"] if n_ref >= 3
               else lm["biscoito_pao_bolo_parcial_2ref_vezes_semana"] if n_ref == 2
               else lm["biscoito_pao_bolo_parcial_1ref_vezes_semana"])
    prep_mes = (lm["preparacoes_doces_integral_vezes_semana"] * SEMANAS_MES
                if periodo == "integral" else lm["preparacoes_doces_parcial_vezes_mes"])
    marg_mes = (lm["margarina_integral_vezes_semana"] * SEMANAS_MES
                if periodo == "integral" else lm["margarina_parcial_vezes_mes"])

    return PerfilNutricional(
        grupo=grupo,
        num_refeicoes=n_ref,
        periodo=periodo,
        pct_cobertura=pct,
        energia_ref=en_ref,
        sodio_max_mg=sodio,
        diversidade_min=div_min,
        flv_min_g_semana=flv_g,
        frutas_dias_min=freq[f"frutas_in_natura_{periodo}"],
        legumes_dias_min=freq[f"legumes_verduras_{periodo}"],
        ferro_dias_min=freq_micro["ferro_heme"],
        vit_a_dias_min=freq_micro["vitamina_a"],
        min_energia=en_ref,
        min_proteina=_lim_min("proteina_g") or pct * 15,
        min_carboidratos=_lim_min("carboidratos_g") or pct * 80,
        min_ferro=_lim_min("ferro_mg") or pct * 4.1,
        min_vitamina_a=_lim_min("vitamina_a_mcg") or pct * 300,
        min_vitamina_c=_lim_min("vitamina_c_mg") or pct * 15,
        min_calcio=_lim_min("calcio_mg"),
        lip_min_g=lip_min,
        lip_max_g=lip_max,
        max_carneos_mes=lm["produtos_carneos_vezes_mes"],
        max_conservas_mes=lm["conservas_vezes_mes"],
        max_doces_mes=lm["doces_vezes_mes"],
        max_lacteos_adic_mes=lac_mes,
        max_biscoito_pao_semana=bis_sem,
        max_prep_doces_mes=prep_mes,
        max_margarina_mes=marg_mes,
        pct_af_meta=restricoes["agricultura_familiar"]["percentual_meta_fnde_2026"],
        pct_af_legal=restricoes["agricultura_familiar"]["percentual_minimo_legal"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRUÇÃO E RESOLUÇÃO DO MODELO MILP
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResultadoSolver:
    """Encapsula os resultados de uma instância do solver."""
    grupo          : str
    mes            : int
    status         : str
    custo_total    : float
    custo_af       : float
    pct_af         : float
    qtd_total_g    : float
    n_alimentos    : int
    tempo_s        : float
    cardapio       : pd.DataFrame = field(default_factory=pd.DataFrame)
    conformidade   : pd.DataFrame = field(default_factory=pd.DataFrame)
    custo_semana   : list         = field(default_factory=list)


def construir_e_resolver(
    df: pd.DataFrame,
    perfil: PerfilNutricional,
    mes: int,
    nome_instancia: str = "",
    time_limit: int = 180,
) -> ResultadoSolver:
    """
    Constrói e resolve o MILP semanal com horizonte de 20 dias úteis.

    Índices:
        j ∈ J = alimentos eligíveis (0..N-1)
        d ∈ D = dias úteis (0..19)
        s ∈ S = semanas (0..3), onde s = d // 5
        ds(s) = dias da semana s = {5s, 5s+1, 5s+2, 5s+3, 5s+4}

    Variáveis:
        x[j,d] ≥ 0      quantidade (g) do alimento j no dia d
        y[j,d] ∈ {0,1}  alimento j ofertado no dia d
        z[j]   ∈ {0,1}  alimento j aparece em algum dia do mês (para diversidade)
    """
    import time as _time

    J = list(df.index)
    N = len(J)
    D = list(range(DIAS_UTEIS_MES))
    S = list(range(SEMANAS_MES))

    # Vetores de coeficientes (valores por 100g — divididos por 100 nas restrições)
    def _vec(col):
        if col in df.columns:
            return df[col].values.astype(float)
        return np.zeros(N)

    c_j   = df["preco_g"].values.astype(float)
    en_j  = _vec("energia")
    prot_j= _vec("proteina")
    carb_j= _vec("carboidratos")
    lip_j = _vec("lipideos")
    sod_j = _vec("sodio")
    fe_j  = _vec("ferro")
    va_j  = _vec("vitamina_a")
    vc_j  = _vec("vitamina_c")
    ca_j  = _vec("calcio")
    af_j  = df["agricultura_familiar"].astype(bool).values

    # Flags de grupo para restrições de frequência
    def _flag(col):
        if col in df.columns:
            return df[col].astype(bool).values
        return np.zeros(N, dtype=bool)

    fl_fruta  = _flag("flag_frutas_in_natura")
    fl_leg    = _flag("flag_legumes_verduras")
    fl_fe     = _flag("flag_ferro_heme")
    fl_va     = _flag("flag_vitamina_a")
    fl_carneo = _flag("flag_produtos_carneos")
    fl_cons   = _flag("flag_conservas")
    fl_doce   = _flag("flag_doces")
    fl_lac    = _flag("flag_lacteos_aditivados")
    fl_bis    = _flag("flag_biscoito_pao_bolo")
    fl_pdoce  = _flag("flag_preparacoes_doces")
    fl_marg   = _flag("flag_margarina_creme_vegetal")
    fl_flv    = fl_fruta | fl_leg
    fl_in     = ~_flag("flag_ultraprocessados")  # in natura / minimamente proc.

    # ── Criar modelo ──────────────────────────────────────────────────────────
    nome = nome_instancia or f"PNAE_{perfil.grupo}_{NOMES_MESES.get(mes,'')}"
    modelo = LpProblem(name=nome, sense=LpMinimize)

    # ── Variáveis ─────────────────────────────────────────────────────────────
    x = [[LpVariable(f"x_{j}_{d}", lowBound=0, upBound=M_BIG)
          for d in D] for j in range(N)]
    y = [[LpVariable(f"y_{j}_{d}", cat="Binary")
          for d in D] for j in range(N)]
    z = [LpVariable(f"z_{j}", cat="Binary") for j in range(N)]

    # ── Função objetivo: minimizar custo total do mês (R$/aluno) ─────────────
    modelo += lpSum(c_j[j] * x[j][d] for j in range(N) for d in D), "Custo_Mensal"

    # ── R01–R07: Restrições nutricionais mínimas (por dia) ────────────────────
    for d in D:
        modelo += lpSum((en_j[j]/100)  * x[j][d] for j in range(N)) >= perfil.min_energia,    f"R01_en_d{d}"
        modelo += lpSum((prot_j[j]/100)* x[j][d] for j in range(N)) >= perfil.min_proteina,   f"R02_prot_d{d}"
        modelo += lpSum((carb_j[j]/100)* x[j][d] for j in range(N)) >= perfil.min_carboidratos,f"R03_carb_d{d}"
        modelo += lpSum((fe_j[j]/100)  * x[j][d] for j in range(N)) >= perfil.min_ferro,      f"R04_ferro_d{d}"
        modelo += lpSum((va_j[j]/100)  * x[j][d] for j in range(N)) >= perfil.min_vitamina_a, f"R05_vita_d{d}"
        modelo += lpSum((vc_j[j]/100)  * x[j][d] for j in range(N)) >= perfil.min_vitamina_c, f"R06_vitc_d{d}"
        if perfil.min_calcio:
            modelo += lpSum((ca_j[j]/100)*x[j][d] for j in range(N)) >= perfil.min_calcio,    f"R07_calc_d{d}"

    # ── R08: Sódio máximo (por dia) ───────────────────────────────────────────
        modelo += lpSum((sod_j[j]/100)* x[j][d] for j in range(N)) <= perfil.sodio_max_mg,    f"R08_sodio_d{d}"

    # ── R09–R10: Lipídeos min/max (por dia — forma absoluta) ──────────────────
        modelo += lpSum((lip_j[j]/100)* x[j][d] for j in range(N)) >= perfil.lip_min_g,       f"R09_lip_min_d{d}"
        modelo += lpSum((lip_j[j]/100)* x[j][d] for j in range(N)) <= perfil.lip_max_g,       f"R10_lip_max_d{d}"

    # ── R11–R12: Vinculação x ↔ y (Big-M) ────────────────────────────────────
    for j in range(N):
        for d in D:
            modelo += x[j][d] <= M_BIG * y[j][d],       f"R11_xley_{j}_{d}"
            modelo += x[j][d] >= EPS_MIN_G * y[j][d],   f"R12_xgey_{j}_{d}"

    # ── R13: Vinculação y → z ─────────────────────────────────────────────────
    for j in range(N):
        for d in D:
            modelo += z[j] >= y[j][d], f"R13_zgy_{j}_{d}"

    # ── R14: Diversidade mínima de alimentos in natura/min. proc. por semana ──
    # Regra: diversidade mínima SEMANAL → aplicada a cada uma das 4 semanas
    # ═════════════════════════════════════════════════════════════════════
    # CORREÇÃO (jul/2026): z_s[j] só tinha limite INFERIOR (z_s >= y[j][d]),
    # que força z_s=1 quando o alimento é usado, mas nada impedia o solver de
    # também colocar z_s=1 para alimentos NUNCA usados naquela semana
    # (y[j][d]=0 em todos os dias). Como z_s não aparece na função objetivo
    # nem em mais nenhuma restrição, a forma mais barata de satisfazer
    # "soma de z_s >= diversidade_min" era simplesmente ligar 23 z_s "no
    # papel", sem custo e sem qualquer alimento real correspondente — a
    # restrição de diversidade era satisfeita apenas formalmente, enquanto o
    # cardápio real (definido por x/y) continuava sendo resolvido só pelas
    # restrições de custo/nutrição, resultando em poucos alimentos distintos
    # (ex.: 6) repetidos todos os dias.
    #
    # Correção: adicionar o limite SUPERIOR que faltava — z_s[j] só pode ser
    # 1 se o alimento for de fato usado em pelo menos um dia daquela semana —
    # fechando a folga que permitia "diversidade fictícia".
    # ═════════════════════════════════════════════════════════════════════
    for s in S:
        dias_s = [s * DIAS_SEMANA + k for k in range(DIAS_SEMANA)]
        # z_s[j] = 1 se alimento j aparece em algum dia da semana s
        z_s = [LpVariable(f"zs_{j}_s{s}", cat="Binary") for j in range(N)]
        for j in range(N):
            for d in dias_s:
                modelo += z_s[j] >= y[j][d], f"R13b_zs_{j}_{s}_{d}"
            modelo += z_s[j] <= lpSum(y[j][d] for d in dias_s), f"R13c_zs_upper_{j}_{s}"
        modelo += (
            lpSum(z_s[j] for j in range(N) if fl_in[j]) >= perfil.diversidade_min,
            f"R14_diversidade_s{s}"
        )

    # ── R15: FLV mínimo semanal (g) ───────────────────────────────────────────
    for s in S:
        dias_s = [s * DIAS_SEMANA + k for k in range(DIAS_SEMANA)]
        modelo += (
            lpSum(x[j][d] for j in range(N) for d in dias_s if fl_flv[j])
            >= perfil.flv_min_g_semana,
            f"R15_flv_s{s}"
        )

    # ── R16–R19: Frequências mínimas semanais de grupos específicos ───────────
    for s in S:
        dias_s = [s * DIAS_SEMANA + k for k in range(DIAS_SEMANA)]

        # Frutas: dias mínimos por semana com pelo menos 1 fruta in natura
        # (garante 'frutas_dias_min' dias fixos da semana com oferta de fruta)
        for d_offset in range(perfil.frutas_dias_min):
            d = dias_s[d_offset]
            modelo += lpSum(y[j][d] for j in range(N) if fl_fruta[j]) >= 1, \
                      f"R16_fruta_s{s}_d{d}"

        # Legumes/verduras: dias mínimos
        for d_offset in range(perfil.legumes_dias_min):
            d = dias_s[d_offset]
            modelo += lpSum(y[j][d] for j in range(N) if fl_leg[j]) >= 1, \
                      f"R17_legumes_s{s}_d{d}"

        # Ferro heme: dias mínimos
        for d_offset in range(perfil.ferro_dias_min):
            d = dias_s[d_offset]
            modelo += lpSum(y[j][d] for j in range(N) if fl_fe[j]) >= 1, \
                      f"R18_ferro_s{s}_d{d}"

        # Vitamina A: dias mínimos
        for d_offset in range(perfil.vit_a_dias_min):
            d = dias_s[d_offset]
            modelo += lpSum(y[j][d] for j in range(N) if fl_va[j]) >= 1, \
                      f"R19_vita_s{s}_d{d}"

    # ── R20–R26: Limites mensais de frequência (20 dias) ─────────────────────
    # Produtos cárneos ≤ 2×/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_carneo[j])
        <= perfil.max_carneos_mes, "R20_carneos_mes"
    )
    # Conservas ≤ 1×/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_cons[j])
        <= perfil.max_conservas_mes, "R21_conservas_mes"
    )
    # Doces ≤ 1×/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_doce[j])
        <= perfil.max_doces_mes, "R22_doces_mes"
    )
    # Lácteos aditivados ≤ limite por período/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_lac[j])
        <= perfil.max_lacteos_adic_mes, "R23_lacteos_mes"
    )
    # Biscoito/pão/bolo ≤ limite por semana
    for s in S:
        dias_s = [s * DIAS_SEMANA + k for k in range(DIAS_SEMANA)]
        modelo += (
            lpSum(y[j][d] for j in range(N) for d in dias_s if fl_bis[j])
            <= perfil.max_biscoito_pao_semana, f"R24_biscoito_s{s}"
        )
    # Preparações doces ≤ limite/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_pdoce[j])
        <= perfil.max_prep_doces_mes, "R25_prep_doces_mes"
    )
    # Margarina ≤ limite/mês
    modelo += (
        lpSum(y[j][d] for j in range(N) for d in D if fl_marg[j])
        <= perfil.max_margarina_mes, "R26_margarina_mes"
    )

    # ── R27: Agricultura Familiar ≥ 45 % do custo total ──────────────────────
    # Forma linear: Σ_{j∈AF}(1-pct)*c_j*x_jd - Σ_{j∉AF} pct*c_j*x_jd ≥ 0
    pct_af = perfil.pct_af_meta
    modelo += (
        lpSum(
            ((1 - pct_af) * c_j[j] if af_j[j] else -pct_af * c_j[j]) * x[j][d]
            for j in range(N) for d in D
        ) >= 0, "R27_AF"
    )

    # ── Resolver ──────────────────────────────────────────────────────────────
    log.info("Resolvendo: %s | mês %02d | %d variáveis | %d restrições",
             perfil.grupo, mes, len(modelo.variables()), len(modelo.constraints))

    t0 = _time.time()
    modelo.solve(PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    tempo = _time.time() - t0

    status = LpStatus[modelo.status]
    log.info("Status: %-12s | tempo: %.1f s", status, tempo)

    if status != "Optimal":
        log.warning("Solução não-ótima para %s mês %02d. Status: %s",
                    perfil.grupo, mes, status)
        return ResultadoSolver(
            grupo=perfil.grupo, mes=mes, status=status,
            custo_total=0, custo_af=0, pct_af=0,
            qtd_total_g=0, n_alimentos=0, tempo_s=tempo,
        )

    # ── Extrair solução ───────────────────────────────────────────────────────
    col_nome = next((c for c in ["alimento","produto_base","produto"] if c in df.columns), df.columns[0])

    linhas = []
    for j in range(N):
        for d in D:
            qty = value(x[j][d]) or 0.0
            if qty < 1e-4:
                continue
            semana = d // DIAS_SEMANA + 1
            dia_semana = d % DIAS_SEMANA + 1
            linhas.append({
                "grupo"      : perfil.grupo,
                "mes"        : mes,
                "nome_mes"   : NOMES_MESES.get(mes, ""),
                "semana"     : semana,
                "dia_semana" : dia_semana,
                "dia_mes"    : d + 1,
                "alimento"   : df.at[j, col_nome],
                "categoria"  : df.at[j, "categoria"] if "categoria" in df.columns else "",
                "qtd_g"      : round(qty, 2),
                "custo_r"    : round(c_j[j] * qty, 5),
                "af"         : bool(af_j[j]),
                "is_fruta"   : bool(fl_fruta[j]),
                "is_legume"  : bool(fl_leg[j]),
                "is_ferro"   : bool(fl_fe[j]),
                "is_vita"    : bool(fl_va[j]),
            })

    df_cardapio = pd.DataFrame(linhas)

    custo_total  = float(value(modelo.objective))
    custo_af     = df_cardapio.loc[df_cardapio["af"], "custo_r"].sum() if not df_cardapio.empty else 0
    pct_af_real  = custo_af / custo_total if custo_total > 0 else 0
    qtd_total    = df_cardapio["qtd_g"].sum() if not df_cardapio.empty else 0
    n_alim       = df_cardapio["alimento"].nunique() if not df_cardapio.empty else 0

    # Custo por semana
    custo_semana = (
        df_cardapio.groupby("semana")["custo_r"].sum().tolist()
        if not df_cardapio.empty else []
    )

    # ── Conformidade nutricional ──────────────────────────────────────────────
    sol = np.zeros((N, DIAS_UTEIS_MES))
    for j in range(N):
        for d in D:
            sol[j, d] = value(x[j][d]) or 0.0

    # CORREÇÃO (jul/2026): quando valor_medio_dia é matematicamente igual ao
    # limite, a comparação `pct_r >= 100` (ou `valor_dia <= limite`) podia
    # falhar por ruído de ponto flutuante típico de solução MILP (o CBC pode
    # retornar, p.ex., 19.999999998 em vez de 20.0 exato), fazendo o relatório
    # marcar ❌ para casos que são, na prática, exatamente 100% de atingimento
    # (ex.: Gorduras mín, Sódio máx, Cálcio). Introduzida tolerância relativa
    # de 1e-6 (compatível com a precisão numérica do CBC) para essa comparação.
    _TOL = 1e-6

    def _ating(vec, minimo, nome, unidade, tipo="min"):
        valor_dia = float(np.dot(vec / 100, sol.mean(axis=1)))  # média diária
        limite    = minimo
        pct_r     = valor_dia / limite * 100 if limite and limite > 0 else None
        if tipo == "max":
            ok = valor_dia <= limite * (1 + _TOL) + _TOL
        else:
            ok = (pct_r or 0) >= 100 * (1 - _TOL) - _TOL
        return {"nutriente": nome, "valor_medio_dia": round(valor_dia, 3),
                "limite": limite, "unidade": unidade,
                "pct_atingido": round(pct_r, 1) if pct_r else "N/D",
                "status": "✅" if ok else "❌"}

    # CORREÇÃO (jul/2026): a conformidade só reportava limites NUTRICIONAIS
    # (energia, macro/micronutrientes, sódio). As restrições de FREQUÊNCIA
    # semanal (frutas in natura ≥ frutas_dias_min dias; legumes/verduras ≥
    # legumes_dias_min dias — R16/R17) eram enforçadas no modelo, mas nunca
    # apareciam no relatório de conformidade, tornando impossível verificar
    # visualmente se estavam de fato sendo cumpridas por semana.
    def _ating_frequencia(flag_vec, dias_min, nome):
        dias_com_item_por_semana = []
        for s in S:
            dias_s = [s * DIAS_SEMANA + k for k in range(DIAS_SEMANA)]
            n_dias_ok = sum(
                1 for d in dias_s
                if any(sol[j, d] > 1e-4 for j in range(N) if flag_vec[j])
            )
            dias_com_item_por_semana.append(n_dias_ok)
        media_dias = float(np.mean(dias_com_item_por_semana)) if dias_com_item_por_semana else 0.0
        ok = all(n >= dias_min for n in dias_com_item_por_semana)
        return {"nutriente": nome,
                "valor_medio_dia": round(media_dias, 2),
                "limite": dias_min, "unidade": "dias/semana",
                "pct_atingido": round(media_dias / dias_min * 100, 1) if dias_min else "N/D",
                "status": "✅" if ok else "❌",
                "detalhe_semanas": dias_com_item_por_semana}

    conf = [
        _ating(en_j,  perfil.min_energia,     "Energia",        "kcal"),
        _ating(prot_j,perfil.min_proteina,     "Proteína",       "g"),
        _ating(carb_j,perfil.min_carboidratos, "Carboidratos",   "g"),
        _ating(lip_j, perfil.lip_min_g,        "Gorduras (mín)", "g"),
        _ating(lip_j, perfil.lip_max_g,        "Gorduras (máx)", "g", "max"),
        _ating(fe_j,  perfil.min_ferro,        "Ferro",          "mg"),
        _ating(va_j,  perfil.min_vitamina_a,   "Vitamina A",     "mcg"),
        _ating(vc_j,  perfil.min_vitamina_c,   "Vitamina C",     "mg"),
        _ating(sod_j, perfil.sodio_max_mg,     "Sódio (máx)",    "mg", "max"),
        _ating_frequencia(fl_fruta, perfil.frutas_dias_min,   "Frequência frutas in natura"),
        _ating_frequencia(fl_leg,   perfil.legumes_dias_min,  "Frequência legumes/verduras"),
    ]
    if perfil.min_calcio:
        conf.append(_ating(ca_j, perfil.min_calcio, "Cálcio", "mg"))

    return ResultadoSolver(
        grupo=perfil.grupo,
        mes=mes,
        status=status,
        custo_total=round(custo_total, 4),
        custo_af=round(custo_af, 4),
        pct_af=round(pct_af_real, 4),
        qtd_total_g=round(qtd_total, 2),
        n_alimentos=n_alim,
        tempo_s=round(tempo, 1),
        cardapio=df_cardapio,
        conformidade=pd.DataFrame(conf),
        custo_semana=custo_semana,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ORQUESTRADOR: TODOS OS GRUPOS × TODOS OS MESES
# ═══════════════════════════════════════════════════════════════════════════════

def resolver_todos(
    dados: dict,
    grupos: Optional[list] = None,
    meses: Optional[list] = None,
    output_folder: str = "./data/output/",
    time_limit: int = 180,
) -> list[ResultadoSolver]:
    """
    Resolve o MILP para todas as combinações (grupo × mês).
    Salva resultados parciais a cada instância resolvida.

    Parameters
    ----------
    dados        : dict retornado por carregar_bases()
    grupos       : lista de grupos a processar (None = todos disponíveis)
    meses        : lista de meses (None = MESES_LETIVOS)
    output_folder: pasta de saída
    time_limit   : tempo máximo por instância (segundos)
    """
    df_raw, col_map = preparar_dataset_solver(dados)
    restricoes = dados["restricoes"]
    df_params  = dados["params"]
    df_saz     = dados.get("sazonalidade", pd.DataFrame())

    grupos_disp = df_params["grupo_nutricional"].unique().tolist()
    grupos_proc = grupos or grupos_disp
    meses_proc  = meses or MESES_LETIVOS

    log.info("Grupos a processar : %s", grupos_proc)
    log.info("Meses a processar  : %s", [NOMES_MESES[m] for m in meses_proc])
    log.info("Total de instâncias: %d", len(grupos_proc) * len(meses_proc))

    resultados = []
    linhas_cardapio_all = []
    linhas_conf_all     = []

    for grupo in grupos_proc:
        try:
            perfil = montar_perfil(grupo, df_params, restricoes)
        except ValueError as e:
            log.error("Grupo '%s' ignorado: %s", grupo, e)
            continue

        for mes in meses_proc:
            # Aplicar sazonalidade ao preço
            df_mes = aplicar_sazonalidade(df_raw, df_saz, mes)

            res = construir_e_resolver(
                df_mes, perfil, mes,
                nome_instancia=f"PNAE_{grupo}_{mes:02d}",
                time_limit=time_limit,
            )
            resultados.append(res)

            if not res.cardapio.empty:
                linhas_cardapio_all.append(res.cardapio)
            if not res.conformidade.empty:
                conf_c = res.conformidade.copy()
                conf_c["grupo"] = grupo
                conf_c["mes"]   = mes
                linhas_conf_all.append(conf_c)

            # Salvar parcial
            _salvar_parcial(res, output_folder)

    # ── Consolidar e salvar arquivos finais ───────────────────────────────────
    if linhas_cardapio_all:
        pd.concat(linhas_cardapio_all, ignore_index=True).to_csv(
            os.path.join(output_folder, "cardapio_mensal_todos_grupos.csv"),
            index=False, encoding="utf-8-sig",
        )
        log.info("✓ cardapio_mensal_todos_grupos.csv")

    if linhas_conf_all:
        pd.concat(linhas_conf_all, ignore_index=True).to_csv(
            os.path.join(output_folder, "conformidade_nutricional_todos.csv"),
            index=False, encoding="utf-8-sig",
        )
        log.info("✓ conformidade_nutricional_todos.csv")

    # Sumário consolidado
    sumario = [{
        "grupo"       : r.grupo,
        "mes"         : r.mes,
        "nome_mes"    : NOMES_MESES.get(r.mes, ""),
        "status"      : r.status,
        "custo_total" : r.custo_total,
        "custo_af"    : r.custo_af,
        "pct_af"      : r.pct_af,
        "pct_af_pct"  : round(r.pct_af * 100, 1),
        "meta_af_ok"  : r.pct_af >= 0.45,
        "n_alimentos" : r.n_alimentos,
        "qtd_total_g" : r.qtd_total_g,
        "tempo_s"     : r.tempo_s,
    } for r in resultados]

    pd.DataFrame(sumario).to_csv(
        os.path.join(output_folder, "sumario_otimizacao_todos.csv"),
        index=False, encoding="utf-8-sig",
    )
    log.info("✓ sumario_otimizacao_todos.csv")

    with open(os.path.join(output_folder, "sumario_otimizacao_todos.json"),
              "w", encoding="utf-8") as f:
        json.dump(sumario, f, ensure_ascii=False, indent=2, default=str)
    log.info("✓ sumario_otimizacao_todos.json")

    n_ok  = sum(1 for r in resultados if r.status == "Optimal")
    n_nok = len(resultados) - n_ok
    log.info("Concluído: %d ótimas | %d não-ótimas", n_ok, n_nok)

    return resultados


def _salvar_parcial(res: ResultadoSolver, output_folder: str):
    """Salva CSV parcial de cada instância para recuperação em caso de falha."""
    if res.cardapio.empty:
        return
    nome = f"cardapio_{res.grupo}_{res.mes:02d}.csv"
    res.cardapio.to_csv(
        os.path.join(output_folder, nome),
        index=False, encoding="utf-8-sig",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Solver MILP do PNAE — cardápio otimizado para todos os grupos e meses."
    )
    parser.add_argument("--output", default="./data/output/",
                        help="Pasta de saída dos módulos anteriores (default: ./data/output/)")
    parser.add_argument("--grupos", nargs="*", default=None,
                        help="Grupos a processar (default: todos). Ex: FUND_6_10 CRECHE")
    parser.add_argument("--meses", nargs="*", type=int, default=None,
                        help="Meses a processar (default: 2-12, jan excluído). Ex: 3 9")
    parser.add_argument("--time-limit", type=int, default=180,
                        help="Tempo máximo por instância em segundos (default: 180)")
    args = parser.parse_args()

    log.info("═" * 60)
    log.info("SOLVER PNAE — iniciando em %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Pasta de saída : %s", args.output)
    log.info("═" * 60)

    dados = carregar_bases(args.output)
    resultados = resolver_todos(
        dados,
        grupos=args.grupos,
        meses=args.meses,
        output_folder=args.output,
        time_limit=args.time_limit,
    )

    log.info("Solver finalizado. Total de instâncias: %d", len(resultados))