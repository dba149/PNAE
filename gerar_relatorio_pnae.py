"""
gerar_relatorio_pnae.py
═══════════════════════════════════════════════════════════════════════════════
Gerador do relatório Word para nutricionistas — dados reais do solver PNAE

Projeto : Otimização inteligente do PNAE
Aluna   : Daniela Barreto Araújo
Orient. : Renato Cividini Matthiesen

USO:
    python gerar_relatorio_pnae.py [--output ./data/output/] [--docx relatorio.docx]

PRÉ-REQUISITOS:
    pip install python-docx pandas

QUANDO ATUALIZAR O RELATÓRIO:
    1. Re-execute o solver:
         python otimizacao_pnae_solver.py --output ./data/output/
    2. Re-execute este script:
         python gerar_relatorio_pnae.py --output ./data/output/
    O Word será regenerado com os dados mais recentes automaticamente.

ARQUIVOS LIDOS (gerados pelo solver):
    cardapio_mensal_todos_grupos.csv   → cardápios diários por grupo/mês
    conformidade_nutricional_todos.csv → conformidade nutricional por instância
    sumario_otimizacao_todos.csv       → indicadores consolidados (custo, AF, status)
    restricoes_pnae.json               → parâmetros dos grupos nutricionais
    parametros_nutricionais_solver.csv → perfis nutricionais dos grupos
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.shared import Pt, RGBColor, Cm, Inches
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relatorio_pnae")

# ── Paleta de cores ───────────────────────────────────────────────────────────
COR_VERDE_FNDE   = RGBColor(0x1A, 0x6B, 0x3C)
COR_AZUL_AF      = RGBColor(0x1A, 0x4F, 0x7A)
COR_PRETO        = RGBColor(0x1A, 0x1A, 0x1A)
COR_CINZA        = RGBColor(0x55, 0x55, 0x55)
COR_AVISO        = RGBColor(0x7B, 0x38, 0x00)

# Hex para shading XML (sem #)
HEX_VERDE_FNDE   = "1A6B3C"
HEX_VERDE_CLARO  = "E8F5EE"
HEX_AZUL_AF      = "1A4F7A"
HEX_AZUL_CLARO   = "E8F0F8"
HEX_BRANCO       = "FFFFFF"
HEX_AVISO_FUNDO  = "FFF3E0"
HEX_OK_FUNDO     = "E8F5EE"
HEX_NOK_FUNDO    = "FFE8E8"

DIAS_SEMANA = {1: "Segunda-feira", 2: "Terça-feira", 3: "Quarta-feira",
               4: "Quinta-feira",  5: "Sexta-feira"}

NOMES_MESES = {
    2: "Fevereiro", 3: "Março",    4: "Abril",    5: "Maio",
    6: "Junho",     7: "Julho",    8: "Agosto",   9: "Setembro",
    10: "Outubro",  11: "Novembro", 12: "Dezembro",
}

LABELS_GRUPOS = {
    "CRECHE"         : "Creche (período integral)",
    "PRE_ESCOLA"     : "Pré-escola",
    "FUND_6_10"      : "Ensino Fundamental I (6–10 anos)",
    "FUND_11_15"     : "Ensino Fundamental II (11–15 anos)",
    "EJA"            : "Educação de Jovens e Adultos (EJA)",
    "FUND_6_10_AEE"  : "Fund. I — Atendimento Educacional Especializado",
    "FUND_11_15_AEE" : "Fund. II — AEE",
    "EJA_AEE"        : "EJA — AEE",
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS DE FORMATAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def _set_cell_bg(cell, hex_color: str):
    """Define cor de fundo de uma célula via XML."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def _set_cell_borders(cell, color="CCCCCC", size="4"):
    """Define bordas finas em uma célula."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    size)
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def _set_row_height(row, height_cm: float):
    """Define altura mínima de linha."""
    trPr = row._tr.get_or_add_trPr()
    trHeight = OxmlElement("w:trHeight")
    trHeight.set(qn("w:val"), str(int(height_cm * 567)))
    trHeight.set(qn("w:hRule"), "atLeast")
    trPr.append(trHeight)


def _paragrafo(doc_or_cell, texto: str, negrito=False, italico=False,
               tamanho=11, cor=None, alinhamento=WD_ALIGN_PARAGRAPH.LEFT,
               espaco_antes=0, espaco_depois=6):
    """Cria parágrafo com formatação completa."""
    if hasattr(doc_or_cell, "add_paragraph"):
        p = doc_or_cell.add_paragraph()
    else:
        p = doc_or_cell.paragraphs[0] if doc_or_cell.paragraphs else doc_or_cell.add_paragraph()
    p.clear()
    p.alignment = alinhamento
    p.paragraph_format.space_before = Pt(espaco_antes)
    p.paragraph_format.space_after  = Pt(espaco_depois)
    run = p.add_run(texto)
    run.bold    = negrito
    run.italic  = italico
    run.font.size = Pt(tamanho)
    run.font.name = "Times New Roman"
    if cor:
        run.font.color.rgb = cor
    return p


def _celula_cabecalho(cel, texto: str, tamanho=10):
    """Célula de cabeçalho verde com texto branco em negrito."""
    _set_cell_bg(cel, HEX_VERDE_FNDE)
    _set_cell_borders(cel, "FFFFFF")
    cel.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cel.paragraphs[0]
    p.clear()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(texto)
    run.bold = True
    run.font.size = Pt(tamanho)
    run.font.name = "Times New Roman"
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _celula_dado(cel, texto: str, fundo=HEX_BRANCO, negrito=False,
                 alinhamento=WD_ALIGN_PARAGRAPH.LEFT, cor=None, tamanho=10):
    """Célula de dado com formatação."""
    _set_cell_bg(cel, fundo)
    _set_cell_borders(cel, "DDDDDD")
    cel.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    p = cel.paragraphs[0]
    p.clear()
    p.alignment = alinhamento
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after  = Pt(1)
    run = p.add_run(str(texto))
    run.bold = negrito
    run.font.size = Pt(tamanho)
    run.font.name = "Times New Roman"
    if cor:
        run.font.color.rgb = cor


def _nota_metodologica(doc, texto: str):
    """Bloco de nota metodológica com fundo amarelo."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    cel = tbl.rows[0].cells[0]
    _set_cell_bg(cel, HEX_AVISO_FUNDO)
    _set_cell_borders(cel, "F0AA40", "6")
    p = cel.paragraphs[0]
    p.clear()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    p.paragraph_format.left_indent  = Pt(6)
    run_label = p.add_run("NOTA METODOLÓGICA  ")
    run_label.bold = True
    run_label.font.size = Pt(10)
    run_label.font.name = "Times New Roman"
    run_label.font.color.rgb = COR_AVISO
    run_texto = p.add_run(texto)
    run_texto.font.size = Pt(10)
    run_texto.font.name = "Times New Roman"
    run_texto.font.color.rgb = COR_AVISO
    doc.add_paragraph()


def _heading1(doc, texto: str):
    p = doc.add_heading(texto, level=1)
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after  = Pt(8)
    for run in p.runs:
        run.font.name  = "Times New Roman"
        run.font.size  = Pt(14)
        run.font.color.rgb = COR_VERDE_FNDE
    # Borda inferior
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "6")
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), HEX_VERDE_FNDE)
    pBdr.append(bottom)
    pPr.append(pBdr)


def _heading2(doc, texto: str):
    p = doc.add_heading(texto, level=2)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after  = Pt(6)
    for run in p.runs:
        run.font.name  = "Times New Roman"
        run.font.size  = Pt(12)
        run.font.color.rgb = COR_AZUL_AF


def _heading3(doc, texto: str):
    p = doc.add_heading(texto, level=3)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after  = Pt(4)
    for run in p.runs:
        run.font.name  = "Times New Roman"
        run.font.size  = Pt(11)
        run.font.color.rgb = COR_PRETO


def _quebra_pagina(doc):
    p = doc.add_paragraph()
    run = p.add_run()
    run.add_break(__import__("docx.enum.text", fromlist=["WD_BREAK"]).WD_BREAK.PAGE)


# ═══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ═══════════════════════════════════════════════════════════════════════════════

def carregar_dados(output_folder: str) -> dict:
    """Carrega e valida todos os CSVs/JSONs necessários."""

    def _csv(nome, obrigatorio=True):
        path = os.path.join(output_folder, nome)
        if not os.path.exists(path):
            if obrigatorio:
                raise FileNotFoundError(
                    f"Arquivo obrigatório não encontrado: {path}\n"
                    f"Execute primeiro: python otimizacao_pnae_solver.py --output {output_folder}"
                )
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False)
        log.info("%-52s → %d linhas × %d colunas", nome, *df.shape)
        return df

    log.info("Carregando dados do solver...")
    dados = {
        "cardapio"     : _csv("cardapio_mensal_todos_grupos.csv"),
        "conformidade" : _csv("conformidade_nutricional_todos.csv"),
        "sumario"      : _csv("sumario_otimizacao_todos.csv"),
        "params"       : _csv("parametros_nutricionais_solver.csv", obrigatorio=False),
    }

    # JSON de restrições (opcional — para parâmetros de perfil)
    json_path = os.path.join(output_folder, "restricoes_pnae.json")
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            dados["restricoes"] = json.load(f)
    else:
        dados["restricoes"] = {}

    # Validações básicas
    assert not dados["cardapio"].empty,     "cardapio_mensal_todos_grupos.csv está vazio"
    assert not dados["conformidade"].empty, "conformidade_nutricional_todos.csv está vazio"
    assert not dados["sumario"].empty,      "sumario_otimizacao_todos.csv está vazio"

    # Derivar grupos e meses presentes nos dados reais
    dados["grupos"] = sorted(dados["cardapio"]["grupo"].unique())
    dados["meses"]  = sorted(dados["cardapio"]["mes"].unique())

    log.info("Grupos encontrados: %s", dados["grupos"])
    log.info("Meses encontrados : %s", [NOMES_MESES.get(m, m) for m in dados["meses"]])
    return dados


# ═══════════════════════════════════════════════════════════════════════════════
# SEÇÕES DO DOCUMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def _capa(doc, dados: dict):
    """Página de capa."""
    for _ in range(6):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("PROGRAMA NACIONAL DE ALIMENTAÇÃO ESCOLAR")
    run.bold = True
    run.font.size = Pt(18)
    run.font.name = "Times New Roman"
    run.font.color.rgb = COR_VERDE_FNDE

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run("CARDÁPIOS OTIMIZADOS POR PROGRAMAÇÃO LINEAR INTEIRA MISTA")
    r2.bold = True
    r2.font.size = Pt(15)
    r2.font.name = "Times New Roman"
    r2.font.color.rgb = COR_AZUL_AF

    doc.add_paragraph()

    for linha in [
        f"Relatório para Nutricionistas e Gestores — Salvador, Bahia",
        f"{len(dados['grupos'])} grupo(s) nutricional(is)  |  "
        f"{len(dados['meses'])} mês(es) letivo(s)  |  20 dias úteis/mês",
        "",
        "Resolução CD/FNDE nº 4, de 26 de fevereiro de 2026  ·  Lei nº 11.947/2009",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(linha)
        run.font.size = Pt(11)
        run.font.name = "Times New Roman"
        if "Resolução" in linha:
            run.italic = True
            run.font.color.rgb = COR_VERDE_FNDE

    doc.add_paragraph()
    doc.add_paragraph()

    for linha in [
        f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "Projeto: Otimização inteligente do PNAE — MBA em Data Science & Analytics",
        "Aluna: Daniela Barreto Araújo  ·  Orientador: Renato Cividini Matthiesen",
    ]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(linha)
        run.font.size = Pt(9)
        run.font.name = "Times New Roman"
        run.font.color.rgb = COR_CINZA

    _quebra_pagina(doc)


def _apresentacao(doc, dados: dict):
    """Seção 1 — Apresentação."""
    _heading1(doc, "1. Apresentação")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    txt = (
        "Este documento apresenta os cardápios mensais otimizados para os grupos nutricionais "
        "atendidos pelas escolas municipais de Salvador — BA, gerados por um modelo de "
        "Programação Linear Inteira Mista (MILP) fundamentado no Problema da Dieta "
        "(Stigler, 1945; Dantzig, 1963) e calibrado conforme a Resolução CD/FNDE nº 4, "
        "de 26 de fevereiro de 2026, e a Lei nº 11.947/2009 (Agricultura Familiar)."
    )
    run = p.add_run(txt)
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p2.paragraph_format.space_after = Pt(8)
    txt2 = (
        "O modelo minimiza o custo de aquisição de alimentos por aluno por refeição, "
        "garantindo simultaneamente: cobertura nutricional mínima por grupo, diversidade "
        "semanal de alimentos in natura, oferta obrigatória de frutas, legumes e verduras, "
        "restrições de frequência mensal de grupos alimentares e cumprimento da meta de "
        "45 % de aquisição da Agricultura Familiar sobre o valor total das compras."
    )
    run2 = p2.add_run(txt2)
    run2.font.size = Pt(11)
    run2.font.name = "Times New Roman"

    _nota_metodologica(
        doc,
        "Janeiro é excluído do planejamento por ser mês de férias escolares no calendário "
        "letivo municipal de Salvador-BA. O planejamento cobre os meses de fevereiro a "
        "dezembro (11 meses letivos × 20 dias úteis = 220 dias por grupo nutricional)."
    )

    # Tabela resumo executivo dos grupos processados
    _heading2(doc, "Grupos nutricionais processados")
    df_sum = dados["sumario"]

    tbl = doc.add_table(rows=1, cols=6)
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl.style = "Table Grid"
    widths = [Cm(4.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(2.5)]
    for i, w in enumerate(widths):
        tbl.columns[i].width = w

    hdrs = ["Grupo", "Meses", "Status", "Custo/mês (R$)", "% AF", "Alimentos"]
    for i, h in enumerate(hdrs):
        _celula_cabecalho(tbl.rows[0].cells[i], h)

    resumo_grupos = df_sum.groupby("grupo").agg(
        meses=("mes", "count"),
        status=("status", lambda x: "Optimal" if (x == "Optimal").all() else "Parcial"),
        custo_medio=("custo_total", "mean"),
        pct_af_medio=("pct_af_pct", "mean"),
        n_alimentos=("n_alimentos", "mean"),
    ).reset_index()

    for i, row in resumo_grupos.iterrows():
        fundo = HEX_VERDE_CLARO if i % 2 == 0 else HEX_BRANCO
        tr = tbl.add_row()
        label = LABELS_GRUPOS.get(row["grupo"], row["grupo"])
        _celula_dado(tr.cells[0], label,                    fundo, negrito=True)
        _celula_dado(tr.cells[1], int(row["meses"]),        fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        ok = row["status"] == "Optimal"
        _celula_dado(tr.cells[2], "✅ Ótimo" if ok else "⚠ Parcial",
                     HEX_OK_FUNDO if ok else HEX_NOK_FUNDO,
                     alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        custo_str = f"R$ {row['custo_medio']:.4f}" if row["custo_medio"] > 0 else "—"
        _celula_dado(tr.cells[3], custo_str,                fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        pct_str  = f"{row['pct_af_medio']:.1f} %" if row["pct_af_medio"] > 0 else "—"
        pct_fundo = HEX_AZUL_CLARO if row["pct_af_medio"] >= 45 else HEX_NOK_FUNDO
        _celula_dado(tr.cells[4], pct_str, pct_fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tr.cells[5], int(row["n_alimentos"]),  fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_paragraph()
    _quebra_pagina(doc)


def _cardapio_grupo_mes(doc, dados: dict, grupo: str, mes: int):
    """Gera a seção de cardápio semanal para um grupo e mês específicos."""
    df_card = dados["cardapio"]
    df_gm   = df_card[(df_card["grupo"] == grupo) & (df_card["mes"] == mes)].copy()

    if df_gm.empty:
        return

    nome_mes   = NOMES_MESES.get(mes, str(mes))
    label_grupo = LABELS_GRUPOS.get(grupo, grupo)
    custo_mes  = df_gm["custo_r"].sum()
    custo_af   = df_gm.loc[df_gm["af"].astype(str).str.lower() == "true", "custo_r"].sum()
    pct_af     = custo_af / custo_mes * 100 if custo_mes > 0 else 0
    n_alim     = df_gm["alimento"].nunique()

    _heading3(doc, f"{nome_mes}  —  {label_grupo}")

    # Linha de indicadores rápidos
    if custo_mes > 0:
        p_ind = doc.add_paragraph()
        p_ind.paragraph_format.space_after = Pt(6)
        for txt, negrito in [
            (f"Custo total do mês: R$ {custo_mes:.4f}/aluno  ", True),
            (f"|  AF: {pct_af:.1f} %  ", False),
            (f"|  {n_alim} alimentos distintos", False),
        ]:
            run = p_ind.add_run(txt)
            run.bold = negrito
            run.font.size = Pt(10)
            run.font.name = "Times New Roman"
            run.font.color.rgb = COR_VERDE_FNDE if negrito else COR_CINZA
    else:
        p_ind = doc.add_paragraph()
        run = p_ind.add_run(
            f"{n_alim} alimentos distintos  "
            "(custo: preços não disponíveis — verifique resumo_precos_referencia.csv)"
        )
        run.italic = True
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"
        run.font.color.rgb = COR_CINZA

    # Uma tabela por semana
    for semana in sorted(df_gm["semana"].unique()):
        df_sem = df_gm[df_gm["semana"] == semana].sort_values(["dia_semana", "alimento"])

        p_sem = doc.add_paragraph()
        p_sem.paragraph_format.space_before = Pt(6)
        p_sem.paragraph_format.space_after  = Pt(3)
        run_sem = p_sem.add_run(f"Semana {semana}")
        run_sem.bold = True
        run_sem.font.size = Pt(10)
        run_sem.font.name = "Times New Roman"
        run_sem.font.color.rgb = COR_AZUL_AF

        # Cabeçalho: Dia | Alimento | Qtd (g) | Custo (R$) | Cat | AF
        tbl = doc.add_table(rows=1, cols=6)
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
        widths = [Cm(2.8), Cm(7.2), Cm(1.5), Cm(1.8), Cm(2.5), Cm(1.2)]
        for i, w in enumerate(widths):
            tbl.columns[i].width = w

        hdrs = ["Dia", "Alimento", "Qtd (g)", "Custo (R$)", "Categoria", "AF"]
        for i, h in enumerate(hdrs):
            _celula_cabecalho(tbl.rows[0].cells[i], h)

        for dia in sorted(df_sem["dia_semana"].unique()):
            df_dia = df_sem[df_sem["dia_semana"] == dia]
            nome_dia = DIAS_SEMANA.get(dia, f"Dia {dia}")

            for idx_item, (_, item) in enumerate(df_dia.iterrows()):
                is_af = str(item.get("af", "")).lower() == "true"
                fundo = HEX_AZUL_CLARO if is_af else (
                    HEX_VERDE_CLARO if dia % 2 == 0 else HEX_BRANCO
                )

                tr  = tbl.add_row()
                _set_row_height(tr, 0.55)

                _celula_dado(tr.cells[0],
                             nome_dia if idx_item == 0 else "",
                             fundo, negrito=(idx_item == 0))
                _celula_dado(tr.cells[1], str(item["alimento"])[:80], fundo)
                _celula_dado(tr.cells[2], f"{item['qtd_g']:.1f}", fundo,
                             alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
                custo_str = f"{item['custo_r']:.4f}" if item["custo_r"] > 0 else "—"
                _celula_dado(tr.cells[3], custo_str, fundo,
                             alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
                cat_str = str(item.get("categoria", "")).replace("_", " ").title()
                _celula_dado(tr.cells[4], cat_str, fundo)
                _celula_dado(tr.cells[5],
                             "✓ AF" if is_af else "—",
                             HEX_AZUL_CLARO if is_af else fundo,
                             negrito=is_af,
                             alinhamento=WD_ALIGN_PARAGRAPH.CENTER,
                             cor=COR_AZUL_AF if is_af else None)

        doc.add_paragraph()


def _secao_cardapios(doc, dados: dict):
    """Parte I — Cardápios mensais para todos os grupos e meses."""
    _heading1(doc, "Parte I — Cardápios Mensais por Grupo Nutricional")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(
        "Cada tabela apresenta a composição diária do cardápio otimizado. "
        "Alimentos destacados em azul são provenientes de fornecedores de "
        "Agricultura Familiar (AF). A coluna Qtd (g) indica a quantidade por "
        "aluno por refeição. As 4 semanas do mês variam os alimentos dentro "
        "dos mesmos grupos nutricionais, garantindo a diversidade mínima exigida "
        "pela Resolução CD/FNDE nº 4/2026, Art. 19."
    )
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    # Legenda de cores
    _heading2(doc, "Legenda")
    tbl_leg = doc.add_table(rows=3, cols=2)
    tbl_leg.style = "Table Grid"
    tbl_leg.columns[0].width = Cm(1.5)
    tbl_leg.columns[1].width = Cm(15)
    dados_leg = [
        (HEX_AZUL_CLARO,  "✓ AF",
         "Alimento de Agricultura Familiar — prioridade de aquisição"),
        (HEX_VERDE_CLARO, "",
         "Linha de alimento convencional (dias alternados)"),
        (HEX_BRANCO,      "",
         "Linha de alimento convencional"),
    ]
    for i, (fundo, label, descricao) in enumerate(dados_leg):
        _celula_dado(tbl_leg.rows[i].cells[0], label, fundo,
                     alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tbl_leg.rows[i].cells[1], descricao, fundo)
    doc.add_paragraph()

    for grupo in dados["grupos"]:
        _heading2(doc, LABELS_GRUPOS.get(grupo, grupo))
        for mes in dados["meses"]:
            _cardapio_grupo_mes(doc, dados, grupo, mes)
        _quebra_pagina(doc)


def _secao_conformidade(doc, dados: dict):
    """Parte II — Conformidade nutricional por grupo e mês."""
    _heading1(doc, "Parte II — Conformidade Nutricional")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(
        "A tabela a seguir confronta os valores nutricionais médios diários atingidos "
        "pelo cardápio otimizado com os limites estabelecidos pela Resolução CD/FNDE "
        "nº 4/2026 para cada grupo nutricional. Valores médios calculados sobre o "
        "horizonte de 20 dias úteis do mês."
    )
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    df_conf = dados["conformidade"]

    for grupo in dados["grupos"]:
        _heading2(doc, LABELS_GRUPOS.get(grupo, grupo))
        df_g = df_conf[df_conf["grupo"] == grupo]

        for mes in sorted(df_g["mes"].unique()):
            df_gm = df_g[df_g["mes"] == mes]
            if df_gm.empty:
                continue

            _heading3(doc, NOMES_MESES.get(mes, str(mes)))

            tbl = doc.add_table(rows=1, cols=5)
            tbl.style = "Table Grid"
            tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
            widths = [Cm(4.5), Cm(3.2), Cm(2.8), Cm(2.3), Cm(4.2)]
            for i, w in enumerate(widths):
                tbl.columns[i].width = w

            hdrs = ["Nutriente", "Valor médio/dia", "Limite FNDE", "% Atingido", "Status"]
            for i, h in enumerate(hdrs):
                _celula_cabecalho(tbl.rows[0].cells[i], h)

            for idx, (_, row) in enumerate(df_gm.iterrows()):
                fundo = HEX_VERDE_CLARO if idx % 2 == 0 else HEX_BRANCO
                ok    = str(row.get("status", "")).strip() == "✅"

                tr = tbl.add_row()
                _celula_dado(tr.cells[0], row["nutriente"],       fundo, negrito=True)
                _celula_dado(tr.cells[1],
                             f"{row['valor_medio_dia']:.3f} {row['unidade']}",
                             fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
                _celula_dado(tr.cells[2],
                             f"{row['limite']} {row['unidade']}",
                             fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
                pct_str = f"{row['pct_atingido']} %" if str(row['pct_atingido']) != "N/D" else "N/D"
                _celula_dado(tr.cells[3], pct_str, fundo,
                             alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
                status_label = "✅ Conforme" if ok else "❌ Não conforme"
                _celula_dado(tr.cells[4], status_label,
                             HEX_OK_FUNDO if ok else HEX_NOK_FUNDO,
                             negrito=True,
                             alinhamento=WD_ALIGN_PARAGRAPH.CENTER,
                             cor=COR_VERDE_FNDE if ok else RGBColor(0xB0, 0x00, 0x20))

            doc.add_paragraph()

    _quebra_pagina(doc)


def _secao_af(doc, dados: dict):
    """Parte III — Indicadores de Agricultura Familiar."""
    _heading1(doc, "Parte III — Indicadores de Agricultura Familiar")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(
        "A Lei nº 11.947/2009 (Art. 14) estabelece que no mínimo 30 % dos recursos "
        "repassados pelo FNDE devem ser utilizados na aquisição de produtos da "
        "Agricultura Familiar. Este projeto adota como meta operacional o percentual "
        "de 45 %, conforme orientação FNDE 2026, incorporado como restrição hard "
        "no modelo de otimização (Restrição R27)."
    )
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    df_sum = dados["sumario"]

    tbl = doc.add_table(rows=1, cols=7)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT
    widths = [Cm(3.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(2.5), Cm(3.0)]
    for i, w in enumerate(widths):
        tbl.columns[i].width = w

    hdrs = ["Grupo", "Mês", "Status solver", "Custo total", "Custo AF", "% AF", "Meta 45 %"]
    for i, h in enumerate(hdrs):
        _celula_cabecalho(tbl.rows[0].cells[i], h)

    for idx, (_, row) in enumerate(df_sum.iterrows()):
        fundo = HEX_VERDE_CLARO if idx % 2 == 0 else HEX_BRANCO
        ok_solver = str(row.get("status", "")) == "Optimal"
        pct_af    = float(row.get("pct_af_pct", 0))
        ok_af     = pct_af >= 45 or bool(row.get("meta_af_ok", False))

        custo_str    = f"R$ {row['custo_total']:.4f}" if row["custo_total"] > 0 else "—"
        custo_af_str = f"R$ {row['custo_af']:.4f}"   if row["custo_af"]    > 0 else "—"
        pct_str      = f"{pct_af:.1f} %"              if pct_af            > 0 else "—"

        tr = tbl.add_row()
        label = LABELS_GRUPOS.get(row["grupo"], row["grupo"])
        _celula_dado(tr.cells[0], label,                      fundo, negrito=True)
        _celula_dado(tr.cells[1], NOMES_MESES.get(int(row["mes"]), str(row["mes"])),
                     fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tr.cells[2], "✅ Ótimo" if ok_solver else "⚠ Parcial",
                     HEX_OK_FUNDO if ok_solver else HEX_NOK_FUNDO,
                     alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tr.cells[3], custo_str,    fundo, alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tr.cells[4], custo_af_str,
                     HEX_AZUL_CLARO if ok_af else fundo,
                     alinhamento=WD_ALIGN_PARAGRAPH.CENTER)
        _celula_dado(tr.cells[5], pct_str,
                     HEX_AZUL_CLARO if ok_af else HEX_NOK_FUNDO,
                     negrito=True, alinhamento=WD_ALIGN_PARAGRAPH.CENTER,
                     cor=COR_AZUL_AF if ok_af else RGBColor(0xB0, 0x00, 0x20))
        _celula_dado(tr.cells[6],
                     "✅ Meta atingida" if ok_af else "❌ Abaixo da meta",
                     HEX_OK_FUNDO if ok_af else HEX_NOK_FUNDO,
                     negrito=True, alinhamento=WD_ALIGN_PARAGRAPH.CENTER,
                     cor=COR_VERDE_FNDE if ok_af else RGBColor(0xB0, 0x00, 0x20))

    doc.add_paragraph()

    _nota_metodologica(
        doc,
        "Os percentuais de AF são calculados sobre o custo de aquisição dos alimentos "
        "incluídos no cardápio otimizado. Alimentos com flag 'is_af = True' oriundos do "
        "matching com a base PGPM/CONAB (confiança alta) e do matching heurístico TF-IDF "
        "(confiança média) compõem o numerador. Custo exibido como '—' indica que "
        "resumo_precos_referencia.csv não foi gerado ou o join de preços falhou — "
        "verifique a execução do Módulo 3."
    )

    _quebra_pagina(doc)


def _secao_instrucoes(doc):
    """Seção final — Instruções operacionais para nutricionistas."""
    _heading1(doc, "Instruções Operacionais para Nutricionistas")

    _heading2(doc, "Como interpretar os cardápios")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(
        "Cada linha do cardápio representa um alimento in natura ou minimamente processado "
        "a ser adquirido e preparado pela equipe de cozinha. A coluna Qtd (g) indica a "
        "quantidade por aluno por refeição. Alimentos marcados com ✓ AF devem ser "
        "adquiridos preferencialmente de fornecedores com DAP ou CAF vigentes."
    )
    run.font.size = Pt(11)
    run.font.name = "Times New Roman"

    _heading2(doc, "Adaptações permitidas sem perda de conformidade")
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run2 = p2.add_run(
        "Substituições de alimentos dentro do mesmo grupo nutricional são permitidas desde "
        "que: (1) o alimento substituto possua composição nutricional equivalente ou "
        "superior; (2) o preço de referência não supere o do alimento substituído em "
        "mais de 20 %; (3) a flag de AF seja mantida quando o original era AF."
    )
    run2.font.size = Pt(11)
    run2.font.name = "Times New Roman"

    _heading2(doc, "Restrições que não podem ser alteradas (Art. 18 e 19)")
    restricoes = [
        ("Gordura trans industrializada",           "Proibida em todos os cardápios"),
        ("Ultraprocessados para ≤ 3 anos",          "Proibidos (Art. 18)"),
        ("Açúcar / mel / adoçantes para ≤ 3 anos", "Proibidos (Art. 18)"),
        ("Produtos cárneos processados",            "Máximo 2 vezes por mês"),
        ("Alimentos em conserva",                   "Máximo 1 vez por mês"),
        ("Doces",                                   "Máximo 1 vez por mês"),
        ("Sódio — período parcial 1 refeição",      "≤ 600 mg por refeição"),
        ("Sódio — período integral 3+ refeições",   "≤ 1.400 mg por dia"),
        ("Diversidade — 1 refeição/dia",            "≥ 10 alimentos in natura/semana"),
        ("Diversidade — 3+ refeições/dia",          "≥ 23 alimentos in natura/semana"),
        ("Agricultura Familiar",                    "≥ 45 % do custo total (meta FNDE 2026)"),
    ]
    tbl = doc.add_table(rows=1, cols=2)
    tbl.style = "Table Grid"
    tbl.columns[0].width = Cm(8)
    tbl.columns[1].width = Cm(9)
    _celula_cabecalho(tbl.rows[0].cells[0], "Restrição")
    _celula_cabecalho(tbl.rows[0].cells[1], "Limite")
    for i, (r, l) in enumerate(restricoes):
        fundo = HEX_VERDE_CLARO if i % 2 == 0 else HEX_BRANCO
        tr = tbl.add_row()
        _celula_dado(tr.cells[0], r, fundo, negrito=True)
        _celula_dado(tr.cells[1], l, fundo)
    doc.add_paragraph()

    _heading2(doc, "Referências normativas")
    refs = [
        "BRASIL. Resolução CD/FNDE nº 4, de 26 de fevereiro de 2026. Brasília: FNDE, 2026.",
        "BRASIL. Lei nº 11.947, de 16 de junho de 2009. Diário Oficial da União, Brasília, 17 jun. 2009.",
        "NEPA/UNICAMP. Tabela Brasileira de Composição de Alimentos (TACO). 4. ed. Campinas, 2011.",
        "FoRC/USP. Tabela Brasileira de Composição de Alimentos (TBCA). Versão 7.2. São Paulo, 2023.",
        "STIGLER, G. J. The cost of subsistence. Journal of Farm Economics, v. 27, n. 2, p. 303-314, 1945.",
        "DANTZIG, G. B. Linear programming and extensions. Princeton: Princeton University Press, 1963.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after  = Pt(4)
        p.paragraph_format.left_indent  = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        run = p.add_run(ref)
        run.font.size = Pt(10)
        run.font.name = "Times New Roman"


# ═══════════════════════════════════════════════════════════════════════════════
# MONTAGEM E EXPORTAÇÃO DO DOCUMENTO
# ═══════════════════════════════════════════════════════════════════════════════

def gerar_documento(output_folder: str, docx_saida: str):
    """Monta e salva o documento Word completo."""

    # Carregar dados reais do solver
    dados = carregar_dados(output_folder)

    # Criar documento
    doc = Document()

    # Configurar página: A4, margens 3/2 cm
    section = doc.sections[0]
    section.page_width  = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin    = Cm(3)
    section.bottom_margin = Cm(2)
    section.left_margin   = Cm(3)
    section.right_margin  = Cm(2)

    # Estilo padrão
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)

    # Montar seções
    log.info("Gerando capa...")
    _capa(doc, dados)

    log.info("Gerando apresentação...")
    _apresentacao(doc, dados)

    log.info("Gerando cardápios (%d grupos × %d meses)...",
             len(dados["grupos"]), len(dados["meses"]))
    _secao_cardapios(doc, dados)

    log.info("Gerando conformidade nutricional...")
    _secao_conformidade(doc, dados)

    log.info("Gerando indicadores de AF...")
    _secao_af(doc, dados)

    log.info("Gerando instruções operacionais...")
    _secao_instrucoes(doc)

    # Salvar
    doc.save(docx_saida)
    kb = os.path.getsize(docx_saida) // 1024
    log.info("═" * 55)
    log.info("✓ Relatório gerado: %s", docx_saida)
    log.info("  Tamanho  : %d KB", kb)
    log.info("  Grupos   : %s", dados["grupos"])
    log.info("  Meses    : %s", [NOMES_MESES.get(m, m) for m in dados["meses"]])
    log.info("═" * 55)


# ═══════════════════════════════════════════════════════════════════════════════
# PONTO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gerador do relatório Word PNAE para nutricionistas."
    )
    parser.add_argument(
        "--output", default="./data/output/",
        help="Pasta com os CSVs gerados pelo solver (default: ./data/output/)"
    )
    parser.add_argument(
        "--docx", default=None,
        help="Caminho do arquivo Word de saída (default: <output>/relatorio_pnae_nutricionistas.docx)"
    )
    args = parser.parse_args()

    docx_saida = args.docx or os.path.join(args.output, "relatorio_pnae_nutricionistas.docx")

    log.info("═" * 55)
    log.info("Gerador de Relatório PNAE — %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("Pasta de dados : %s", args.output)
    log.info("Saída Word     : %s", docx_saida)
    log.info("═" * 55)

    gerar_documento(args.output, docx_saida)
