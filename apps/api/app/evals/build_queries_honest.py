"""Genera el fixture honesto de 150 preguntas conservando las 30 originales."""
import json
from pathlib import Path

CT = "v_6c6b9c546b57"      # codigo del trabajo
L822 = "v_c6817dcb363c"    # ley 822
REG = "v_bf61dac6fb4b"     # reglamento issdhu (reingerido con OCR es)
NIIF = "v_d866fe737555"   # reingerido por lotes: secciones 16-29 recuperadas
GAFI = "v_ab926b914066"
A202, A207, A208 = "v_5dc41b3b1b7e", "v_e5954d44aa6b", "v_30f92b80bc2a"

# (id, pregunta, corpus, version, unidades_esperadas, aceptables, nota)
UNITS = [
    # ---------- CODIGO DEL TRABAJO (18 nuevas) ----------
    ("h031", "¿Cuántas horas puede durar la jornada diurna?", "codigo_del_trabajo", CT, ["Arto. 51"], ["Arto. 49", "Arto. 53"], ""),
    ("h032", "¿Cuántas horas extras como máximo puedo hacer?", "codigo_del_trabajo", CT, ["Arto. 58"], ["Arto. 57"], ""),
    ("h033", "¿Cuáles son los días feriados con derecho a descanso?", "codigo_del_trabajo", CT, ["Arto. 66"], ["Arto. 67", "Arto. 68"], ""),
    ("h034", "¿Cada cuánto me toca un día de descanso?", "codigo_del_trabajo", CT, ["Arto. 64"], ["Arto. 65"], "El texto dice 'por cada seis dias de trabajo continuo'."),
    ("h035", "¿En qué fecha me tienen que pagar el aguinaldo?", "codigo_del_trabajo", CT, ["Arto. 95"], ["Arto. 93", "Arto. 94"], "Sinonimo popular de decimo tercer mes."),
    ("h036", "¿Me pueden embargar el décimo tercer mes?", "codigo_del_trabajo", CT, ["Arto. 97"], ["Arto. 92"], ""),
    ("h037", "¿Puedo negarme a trabajar horas extraordinarias?", "codigo_del_trabajo", CT, ["Arto. 59"], ["Arto. 60"], ""),
    ("h038", "¿Qué es el salario mínimo?", "codigo_del_trabajo", CT, ["Arto. 85"], ["Arto. 82"], ""),
    ("h039", "¿Cómo y cuándo me deben pagar el salario?", "codigo_del_trabajo", CT, ["Arto. 86"], ["Arto. 87"], ""),
    ("h040", "¿Qué descuentos me pueden hacer del sueldo?", "codigo_del_trabajo", CT, ["Arto. 88"], ["Arto. 90"], ""),
    ("h041", "¿Puedo pedir permiso si tengo un familiar grave a mi cargo?", "codigo_del_trabajo", CT, ["Arto. 75"], ["Arto. 73", "Arto. 74"], ""),
    ("h042", "¿La empresa tiene que darme equipo de protección?", "codigo_del_trabajo", CT, ["Arto. 103"], ["Arto. 100", "Arto. 101"], ""),
    ("h043", "¿Qué medidas de seguridad debe tomar mi empleador?", "codigo_del_trabajo", CT, ["Arto. 100"], ["Arto. 101", "Arto. 104"], ""),
    ("h044", "¿Me pueden cambiar de puesto sin mi consentimiento?", "codigo_del_trabajo", CT, ["Arto. 31"], ["Arto. 32", "Arto. 33"], ""),
    ("h045", "¿Qué significa que se suspenda el contrato de trabajo?", "codigo_del_trabajo", CT, ["Arto. 35"], ["Arto. 36", "Arto. 37"], ""),
    ("h046", "Si quiero renunciar, ¿con cuánto tiempo debo avisar?", "codigo_del_trabajo", CT, ["Arto. 44"], ["Arto. 43"], "El texto no usa la palabra 'preaviso'."),
    ("h047", "¿Las embarazadas tienen una jornada distinta?", "codigo_del_trabajo", CT, ["Arto. 52"], ["Arto. 141"], ""),
    ("h048", "¿Quiénes no tienen límite de jornada laboral?", "codigo_del_trabajo", CT, ["Arto. 61"], [], "Gerentes y apoderados."),
    # ---------- LEY 822 (26 nuevas) ----------
    ("h049", "¿Cuándo se considera que alguien es residente para pagar impuestos?", "ley_822", L822, ["Arto. 7"], [], ""),
    ("h050", "¿Qué se considera un establecimiento permanente?", "ley_822", L822, ["Arto. 8"], [], ""),
    ("h051", "¿Qué es un paraíso fiscal según la ley?", "ley_822", L822, ["Arto. 9"], [], ""),
    ("h052", "¿Qué ingresos se consideran de fuente nicaragüense?", "ley_822", L822, ["Arto. 10"], [], ""),
    ("h053", "¿Qué se considera renta del trabajo?", "ley_822", L822, ["Arto. 11"], ["Arto. 12"], ""),
    ("h054", "¿Qué son las rentas de actividades económicas?", "ley_822", L822, ["Arto. 13"], ["Arto. 14"], ""),
    ("h055", "¿Qué son las rentas de capital?", "ley_822", L822, ["Arto. 15"], ["Arto. 16"], ""),
    ("h056", "¿Qué instituciones están exentas de pagar impuesto?", "ley_822", L822, ["Arto. 32"], ["Arto. 33"], ""),
    ("h057", "¿Qué gastos puede deducirse una empresa?", "ley_822", L822, ["Arto. 39"], ["Arto. 42"], ""),
    ("h058", "¿Qué gastos no se pueden deducir?", "ley_822", L822, ["Arto. 43"], ["Arto. 41"], ""),
    ("h059", "¿Cómo se valúan los inventarios para efectos fiscales?", "ley_822", L822, ["Arto. 44"], [], ""),
    ("h060", "¿Cómo se deprecian los activos?", "ley_822", L822, ["Arto. 45"], [], ""),
    ("h061", "¿Qué se entiende por renta bruta?", "ley_822", L822, ["Arto. 36"], ["Arto. 37"], ""),
    ("h062", "¿Cómo se determina la base imponible?", "ley_822", L822, ["Arto. 35"], ["Arto. 36"], ""),
    ("h063", "¿Qué requisitos debe cumplir un gasto para ser deducible?", "ley_822", L822, ["Arto. 42"], ["Arto. 39"], ""),
    ("h064", "¿Se pueden deducir las donaciones?", "ley_822", L822, ["Arto. 40"], [], ""),
    ("h065", "¿Qué principios rigen los tributos en Nicaragua?", "ley_822", L822, ["Arto. 2"], [], ""),
    ("h066", "¿A quiénes se les aplica esta ley tributaria?", "ley_822", L822, ["Arto. 4"], ["Arto. 5"], ""),
    ("h067", "¿En qué territorio se aplica el impuesto?", "ley_822", L822, ["Arto. 5"], ["Arto. 4"], ""),
    ("h068", "¿Cómo se valoran las operaciones entre empresas relacionadas?", "ley_822", L822, ["Arto. 6"], [], ""),
    ("h069", "¿Qué condiciones hay que cumplir para mantener una exención?", "ley_822", L822, ["Arto. 33"], ["Arto. 32"], ""),
    ("h070", "¿Qué ingresos quedan fuera del impuesto?", "ley_822", L822, ["Arto. 34"], ["Arto. 37"], ""),
    ("h071", "¿Qué grava el impuesto sobre la renta?", "ley_822", L822, ["Arto. 3"], ["Arto. 2"], ""),
    ("h072", "¿Cómo se integran las rentas de capital con las demás?", "ley_822", L822, ["Arto. 38"], ["Arto. 15"], ""),
    ("h073", "¿Hay límites a lo que puedo deducir?", "ley_822", L822, ["Arto. 41"], ["Arto. 43"], ""),
    ("h074", "¿Cuándo un ingreso del trabajo tiene vínculo económico con el país?", "ley_822", L822, ["Arto. 12"], ["Arto. 11"], ""),
    # ---------- REGLAMENTO ISSDHU (18 nuevas) ----------
    ("h075", "¿Cómo se paga la cotización si estoy afiliado voluntario?", "reglamento_issdhu", REG, ["Arto. 21"], ["Arto. 19", "Arto. 20"], ""),
    ("h076", "¿Cuál es la pensión mínima que puedo recibir?", "reglamento_issdhu", REG, ["Arto. 51"], ["Arto. 64"], ""),
    ("h077", "¿Cómo se calcula el monto de la pensión de vejez?", "reglamento_issdhu", REG, ["Arto. 52"], ["Arto. 50", "Arto. 53"], ""),
    ("h078", "¿Hay un tope máximo para la pensión?", "reglamento_issdhu", REG, ["Arto. 53"], ["Arto. 51"], ""),
    ("h079", "¿Qué es la fe de vida y para qué sirve?", "reglamento_issdhu", REG, ["Arto. 61"], ["Arto. 62"], ""),
    ("h080", "¿Por qué motivos me pueden suspender la pensión?", "reglamento_issdhu", REG, ["Arto. 72"], ["Arto. 73", "Arto. 62"], ""),
    ("h081", "¿Cuándo se pierde definitivamente la pensión?", "reglamento_issdhu", REG, ["Arto. 73"], ["Arto. 72", "Arto. 74"], ""),
    ("h082", "¿Las pensiones se ajustan con el tiempo?", "reglamento_issdhu", REG, ["Arto. 63"], ["Arto. 64"], ""),
    ("h083", "¿Qué se considera accidente de trabajo?", "reglamento_issdhu", REG, ["Arto. 80"], ["Arto. 78"], ""),
    ("h084", "¿Cómo se reporta un accidente laboral?", "reglamento_issdhu", REG, ["Arto. 81"], ["Arto. 83"], ""),
    ("h085", "¿Existe ayuda para gastos de entierro?", "reglamento_issdhu", REG, ["Arto. 89"], ["Arto. 88"], "El texto dice 'auxilio funerario'."),
    ("h086", "¿Dan algún apoyo por lactancia?", "reglamento_issdhu", REG, ["Arto. 92"], ["Arto. 91", "Arto. 93"], ""),
    ("h087", "¿De cuánto es el subsidio de lactancia?", "reglamento_issdhu", REG, ["Arto. 94"], ["Arto. 92"], ""),
    ("h088", "¿El instituto otorga préstamos a los afiliados?", "reglamento_issdhu", REG, ["Arto. 100"], ["Arto. 101", "Arto. 102"], ""),
    ("h089", "¿Hay programas de vivienda para afiliados?", "reglamento_issdhu", REG, ["Arto. 103"], ["Arto. 104"], ""),
    ("h090", "¿Qué es el plan voluntario de ahorro?", "reglamento_issdhu", REG, ["Arto. 105"], [], ""),
    ("h091", "¿Cubren prótesis u órtesis?", "reglamento_issdhu", REG, ["Arto. 96"], ["Arto. 97"], ""),
    ("h092", "¿Qué papeles necesito para tramitar mi pensión?", "reglamento_issdhu", REG, ["Arto. 75"], ["Arto. 76"], ""),
    # ---------- NIIF PARA PYMES (25 nuevas) ----------
    ("h093", "¿Qué se considera una pequeña o mediana entidad?", "niif_pymes", NIIF, ["Seccion 1"], [], ""),
    ("h094", "¿Qué debe mostrar el estado de situación financiera?", "niif_pymes", NIIF, ["Seccion 4"], ["Seccion 3"], ""),
    ("h095", "¿Cómo se prepara el estado de flujos de efectivo?", "niif_pymes", NIIF, ["Seccion 7"], [], ""),
    ("h096", "¿Qué información va en las notas a los estados financieros?", "niif_pymes", NIIF, ["Seccion 8"], [], ""),
    ("h097", "¿Cómo se miden y contabilizan los inventarios?", "niif_pymes", NIIF, ["Seccion 13"], [], ""),
    ("h098", "¿Cómo se registran las inversiones en asociadas?", "niif_pymes", NIIF, ["Seccion 14"], ["Seccion 15"], ""),
    # Restauradas: las secciones 16-29 volvieron al indice al corregir la
    # extraccion por lotes (antes docling las descartaba por falta de memoria).
    ("h151", "¿Qué son las propiedades de inversión?", "niif_pymes", NIIF, ["Seccion 16"], ["Seccion 17"], "Restaurada tras recuperar las secciones ausentes."),
    ("h152", "¿Cómo se deprecia la propiedad, planta y equipo?", "niif_pymes", NIIF, ["Seccion 17"], ["Seccion 27"], "Restaurada."),
    ("h153", "¿Cómo se contabiliza un arrendamiento?", "niif_pymes", NIIF, ["Seccion 20"], [], "Restaurada."),
    ("h154", "¿Cuándo se debe reconocer una provisión?", "niif_pymes", NIIF, ["Seccion 21"], [], "Restaurada."),
    ("h155", "¿Cuándo se reconocen los ingresos por ventas?", "niif_pymes", NIIF, ["Seccion 23"], [], "Restaurada. Seccion 23 tenia cero parrafos en el indice viejo."),
    ("h156", "¿Cómo se contabilizan las prestaciones a los empleados?", "niif_pymes", NIIF, ["Seccion 28"], [], "Restaurada."),
    ("h157", "¿Cómo se registra el impuesto a las ganancias?", "niif_pymes", NIIF, ["Seccion 29"], [], "Restaurada."),
    ("h158", "¿Cómo se determina si un activo perdió valor?", "niif_pymes", NIIF, ["Seccion 27"], [], "Restaurada. El texto dice 'deterioro del valor'."),
    ("h159", "¿Cómo se contabiliza la compra de otra empresa?", "niif_pymes", NIIF, ["Seccion 19"], [], "Restaurada. El texto dice 'combinaciones de negocios'."),
    ("h160", "¿Cómo se distingue un pasivo del patrimonio?", "niif_pymes", NIIF, ["Seccion 22"], [], "Restaurada."),
    ("h161", "¿Cómo se registran los pagos con acciones a empleados?", "niif_pymes", NIIF, ["Seccion 26"], [], "Restaurada."),
    ("h099", "¿Qué se considera un activo según la norma?", "niif_pymes", NIIF, ["Seccion 2"], [], ""),
    ("h100", "¿Qué comprende un juego completo de estados financieros?", "niif_pymes", NIIF, ["Seccion 3"], ["Seccion 8"], ""),
    ("h101", "¿Qué debe mostrar el estado de resultados?", "niif_pymes", NIIF, ["Seccion 5"], ["Seccion 4"], ""),
    ("h102", "¿Cómo se presentan los cambios en el patrimonio?", "niif_pymes", NIIF, ["Seccion 6"], ["Seccion 5"], ""),
    ("h103", "¿Qué instrumentos financieros no se consideran básicos?", "niif_pymes", NIIF, ["Seccion 12"], ["Seccion 11"], ""),
    ("h104", "¿Cómo se contabiliza un negocio conjunto?", "niif_pymes", NIIF, ["Seccion 15"], ["Seccion 14"], ""),
    ("h105", "¿Qué reglas aplican a las actividades agrícolas?", "niif_pymes", NIIF, ["Seccion 34"], [], ""),
    ("h106", "¿Cómo se convierten las operaciones en moneda extranjera?", "niif_pymes", NIIF, ["Seccion 30"], [], ""),
    ("h107", "¿Qué hay que revelar sobre partes relacionadas?", "niif_pymes", NIIF, ["Seccion 33"], [], ""),
    ("h108", "¿Qué se hace con hechos ocurridos después del cierre?", "niif_pymes", NIIF, ["Seccion 32"], [], ""),
    ("h109", "¿Cómo se corrige un error contable de un año anterior?", "niif_pymes", NIIF, ["Seccion 10"], [], ""),
    ("h110", "¿Qué son los instrumentos financieros básicos?", "niif_pymes", NIIF, ["Seccion 11"], ["Seccion 12"], ""),
    ("h111", "¿Cuándo hay que consolidar estados financieros?", "niif_pymes", NIIF, ["Seccion 9"], [], ""),
    ("h112", "¿Qué porcentaje de trabajadores nicaragüenses debe contratar una empresa?", "codigo_del_trabajo", CT, ["Arto. 14"], [], ""),
    ("h113", "¿Qué se hace cuando hay hiperinflación?", "niif_pymes", NIIF, ["Seccion 31"], [], ""),
    ("h114", "¿Cuándo puede celebrarse un contrato de trabajo verbal?", "codigo_del_trabajo", CT, ["Arto. 24"], ["Arto. 23"], ""),
    ("h115", "¿Qué ingresos no forman parte de la renta bruta?", "ley_822", L822, ["Arto. 37"], ["Arto. 34"], ""),
    ("h116", "¿Cuándo una renta de capital tiene vínculo económico con el país?", "ley_822", L822, ["Arto. 16"], ["Arto. 15"], ""),
    ("h117", "¿Qué hay que hacer al aplicar la norma por primera vez?", "niif_pymes", NIIF, ["Seccion 35"], [], "El texto dice 'transicion a la NIIF'."),
]

# (id, pregunta, corpus, version, bloque_esperado, nota)
BLOCKS = [
    # ---------- GAFI (22) ----------
    ("h118", "¿Qué significa aplicar un enfoque basado en riesgo?", "gafi", GAFI, "Evaluación de riesgos y aplicación de un enfoque basado en riesgo", ""),
    ("h119", "¿Qué debe hacer un banco para conocer a su cliente?", "gafi", GAFI, "Debida diligencia del cliente", "El termino tecnico es 'debida diligencia'."),
    ("h120", "¿Cuánto tiempo hay que conservar los registros de operaciones?", "gafi", GAFI, "Mantenimiento de registros", ""),
    ("h121", "¿Qué son las personas expuestas políticamente?", "gafi", GAFI, "Personas expuestas políticamente", ""),
    ("h122", "¿Qué controles aplican a la banca corresponsal?", "gafi", GAFI, "Banca corresponsal", ""),
    ("h123", "¿Cómo se regulan las remesas y transferencias de dinero?", "gafi", GAFI, "Servicios de transferencia de dinero o valores", ""),
    ("h124", "¿Qué se exige frente a las nuevas tecnologías financieras?", "gafi", GAFI, "Nuevas tecnologías", ""),
    ("h125", "¿Cuándo hay que reportar una operación sospechosa?", "gafi", GAFI, "Reporte de operaciones sospechosas", ""),
    ("h126", "¿Puedo avisarle al cliente que lo reporté?", "gafi", GAFI, "Revelación", "Tipping-off. La pregunta evita el termino tecnico."),
    ("h127", "¿Qué obligaciones tienen los abogados y contadores?", "gafi", GAFI, "APNFD: debida diligencia del cliente", "APNFD = actividades y profesiones no financieras designadas."),
    ("h128", "¿Cómo se identifica al dueño real de una empresa?", "gafi", GAFI, "Transparencia y beneficiario final de las personas jurídicas", "El termino tecnico es 'beneficiario final'."),
    ("h129", "¿Quién supervisa a las instituciones financieras?", "gafi", GAFI, "Regulación y supervisión de las instituciones financieras", ""),
    ("h130", "¿Qué es una unidad de inteligencia financiera?", "gafi", GAFI, "Unidades de inteligencia financiera", ""),
    ("h131", "¿Qué controles hay para el traslado de efectivo entre países?", "gafi", GAFI, "Transporte de efectivo", ""),
    ("h132", "¿Qué sanciones deben existir por incumplir?", "gafi", GAFI, "Sanciones", ""),
    ("h133", "¿Cómo se pide ayuda legal a otro país?", "gafi", GAFI, "Asistencia legal mutua", ""),
    ("h134", "¿Se puede extraditar por lavado de activos?", "gafi", GAFI, "Extradición", ""),
    ("h135", "¿Qué medidas aplican con países de mayor riesgo?", "gafi", GAFI, "Países de mayor riesgo", ""),
    ("h136", "¿Qué controles internos debe tener un grupo financiero?", "gafi", GAFI, "Controles internos y filiales y subsidiarias", ""),
    ("h137", "¿Se puede delegar en terceros la identificación del cliente?", "gafi", GAFI, "Dependencia en terceros", ""),
    ("h138", "¿Qué medidas hay para las ONG y fundaciones?", "gafi", GAFI, "Organizaciones sin fines de lucro", "El texto dice 'organizaciones sin fines de lucro'."),
    ("h139", "¿El secreto bancario impide aplicar estas medidas?", "gafi", GAFI, "Leyes sobre el secreto de las instituciones financieras", ""),
    # ---------- ACTAS (11) ----------
    ("h140", "¿Cuántos cotizantes se necesitan para sostener la nómina de pensionados?", "actas_consejo", A208, "sostener la nómina", ""),
    ("h141", "¿Hasta cuándo alcanza el horizonte de vida del régimen?", "actas_consejo", A208, "Horizonte de vida", ""),
    ("h142", "¿Qué se aprobó del presupuesto para 2026?", "actas_consejo", A208, "presupuesto del ISSDHU para el año 2026", ""),
    ("h143", "¿Qué se resolvió sobre la cuenta por cobrar de ENATREL?", "actas_consejo", A208, "Cuenta por cobrar ENATREL", ""),
    ("h144", "¿Se ajustó el aporte laboral de la rama DVM?", "actas_consejo", A208, "ajuste del aporte laboral", ""),
    ("h145", "¿Cómo cerraron los activos totales del instituto?", "actas_consejo", A207, "Activos", ""),
    ("h146", "¿De cuánto fue la cartera total de préstamos?", "actas_consejo", A207, "Cartera total de préstamos", ""),
    ("h147", "¿Cómo va la ejecución del presupuesto?", "actas_consejo", A207, "Ejecución presupuestaria", ""),
    ("h148", "¿Qué dijo la auditoría externa del instituto?", "actas_consejo", A202, "Auditoría Externa", ""),
    ("h149", "¿Qué se acordó sobre las escrituras de Colinas del Marañón?", "actas_consejo", A202, "Escrituras de compraventa", ""),
    ("h150", "¿Qué se propuso para los préstamos cancelados?", "actas_consejo", A202, "préstamos cancelados", ""),
]

rows = []
for cid, q, corpus, vid, units, acceptable, note in UNITS:
    rows.append({
        "id": cid, "question": q, "corpus": corpus, "version_id": vid,
        "expected_granularity": "single_unit", "expected_units": units,
        "acceptable_units": acceptable, "expected_block": "",
        "review_status": "proposed_auto", "notes": note,
    })
for cid, q, corpus, vid, block, note in BLOCKS:
    rows.append({
        "id": cid, "question": q, "corpus": corpus, "version_id": vid,
        "expected_granularity": "small_block", "expected_units": [],
        "acceptable_units": [], "expected_block": block,
        "review_status": "proposed_auto", "notes": note,
    })

base = Path("apps/api/app/evals/queries_honest_v2.jsonl")
existing = [json.loads(line) for line in base.read_text(encoding="utf-8").splitlines() if line.strip()]
seen = {r["id"] for r in existing}
new = [r for r in rows if r["id"] not in seen]

out = Path("apps/api/app/evals/queries_honest_v3.jsonl")
with out.open("w", encoding="utf-8") as fh:
    for r in existing + new:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
total = existing + new
print(f"conservadas: {len(existing)} | nuevas: {len(new)} | TOTAL: {len(total)}")
for corpus, n in sorted(Counter(r["corpus"] for r in total).items()):
    print(f"  {corpus:22} {n}")
