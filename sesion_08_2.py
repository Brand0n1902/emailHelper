import langchain_core
import os
from dotenv import load_dotenv

# Carga GOOGLE_API_KEY desde el archivo .env (en la misma carpeta del script)
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit(
        "Falta GOOGLE_API_KEY. Créala en el archivo .env (ver .env.example)."
    )

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

# Modelo configurable desde .env (GEMINI_MODEL). Si uno está saturado (503/504),
# cambia el valor en .env sin tocar el código. Reintentos más pacientes ante alta demanda.
MODELO = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_KW = dict(model_provider="google_genai", timeout=120, max_retries=6)

# 1. Configurar los prompts
pelicula_prompt = ChatPromptTemplate.from_template(
    "Devuelve solo el nombre de una película del género: {genero_pelicula}"
)
explain_prompt = ChatPromptTemplate.from_template(
    "Genera la descripción de la película: {pelicula}"
)

# 2. Instanciar el modelo

llm = init_chat_model(MODELO, **GEMINI_KW)

# 3. Construir el workflow (cadena) usando el operador | (LCEL nativo)
# En lugar de usar RunnableSequence explícito, encadenamos los componentes.
# Usamos StrOutputParser() para extraer automáticamente el string del AIMessage.
chain = (
    {"pelicula": pelicula_prompt | llm | StrOutputParser()}
    | explain_prompt
    | llm
    | StrOutputParser()
)

# 4. Ejecutar
result = chain.invoke({"genero_pelicula": "ciencia ficción"})
print(result)

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Inicialización moderna (agnóstica al proveedor)
# Usamos temperature=0 para respuestas consistentes en procesos de negocio
modelo = init_chat_model(MODELO, **GEMINI_KW, temperature=0)

# 2. Definición de Nodos Lógicos (Prompts)
# Paso A: Extraer información clave
analisis_prompt = ChatPromptTemplate.from_template(
    "Eres un analista de soporte técnico. Analiza el siguiente correo del cliente.\n"
    "Devuelve una sola línea con este formato: [Sentimiento] - [Problema Principal]\n\n"
    "Correo del cliente: {correo}"
)

# Paso B: Redactar la solución
redaccion_prompt = ChatPromptTemplate.from_template(
    "Eres un asistente de atención al cliente experto en retención.\n"
    "Basado en el siguiente análisis del caso: '{analisis}'\n"
    "Redacta un borrador de respuesta empática y profesional de máximo 3 líneas ofreciendo ayuda."
)

# 3. Construcción del Workflow Lineal (LCEL)
# Lee el flujo de arriba hacia abajo:
workflow_soporte = (
    # Ejecuta el Paso A y guarda el resultado en la variable "analisis"
    {"analisis": analisis_prompt | modelo | StrOutputParser()}
    # Pasa el "analisis" al Paso B
    | redaccion_prompt
    | modelo
    | StrOutputParser()
)

# Ejecución en Producción
correo_entrante = "¡Esto es inaceptable! Mi pedido número #5542 debió llegar el martes. Ya es viernes, la caja llegó aplastada y nadie me da razón."

print("Procesando correo...\n")
borrador_final = workflow_soporte.invoke({"correo": correo_entrante})

print("--- Borrador Generado para el Agente Humano ---")
print(borrador_final)

"""### El Problema de Negocio (El "Dolor")
En empresas de comercio electrónico, logística o servicios, los centros de atención al cliente (Contact Centers) sufren de un problema grave de escalabilidad. Cuando un cliente envía un correo de queja, el proceso manual es lento y costoso:

El agente lee un correo largo y muchas veces emocional.

Abre el sistema (CRM/ERP) para buscar manualmente el número de pedido.

Revisa el estado del envío.

Consulta el manual de políticas de la empresa para saber qué puede ofrecerle al cliente (ej. un reembolso o solo una disculpa).

Redacta el correo desde cero, buscando el tono adecuado e insertando su firma.

Esto genera cuellos de botella, tiempos de respuesta de varios días y un alto nivel de agotamiento en los empleados.

### La Solución: El Workflow Automatizado (El Código)
El código que construimos actúa como un "Copiloto" invisible que hace todo el trabajo pesado en milisegundos antes de que el agente humano siquiera abra el correo.

Así es como la cadena lineal resuelve el problema paso a paso:

1. Triage y Extracción Estructurada (Eslabón 1)
El LLM lee el correo caótico del cliente y usa Pydantic para extraer únicamente los "datos duros" (el ID del pedido) y el contexto emocional (el nivel de frustración). Convierte texto libre e impredecible en variables limpias que un sistema informático puede entender.

2. Enriquecimiento de Datos / Integración de Sistemas (Eslabón 2)
Con el ID del pedido extraído, el sistema no inventa una respuesta, sino que "habla" con el mundo real.
Llama a la API de Logística (que en la vida real sería SAP, Shopify o Salesforce) para ver dónde está el paquete.
Llama a la API de Recursos Humanos para obtener los datos oficiales del agente en turno.

3. Aplicación de la Política de Negocio (Eslabón 3)
Aquí entra el "cerebro" gerencial. El sistema cruza la frustración del cliente con la realidad del envío para dictaminar una solución basada en las reglas de la empresa. Así se asegura que no se ofrezcan reembolsos que la empresa no puede pagar.

4. Redacción Hiper-Personalizada (Eslabón 4)
Finalmente, el sistema redacta un correo impecable. No es un correo genérico; incluye el nombre real del cliente (sacado de la base de datos, no del correo), la resolución exacta que dictó el gerente virtual y la firma corporativa correcta del agente.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuración Inicial
modelo = init_chat_model(MODELO, **GEMINI_KW, temperature=0)


# --- ESLABÓN 1: Extractor ---
class InfoPedido(BaseModel):
    id_pedido: str = Field(
        description="ID del pedido (ej. ORD-123). Si no hay, devuelve 'SIN_ID'."
    )
    emocion: str = Field(description="Nivel de frustración: 'Alta', 'Media' o 'Baja'.")


extractor_prompt = ChatPromptTemplate.from_template(
    "Extrae la información clave del siguiente correo del cliente:\n\n{correo}"
)
extractor = extractor_prompt | modelo.with_structured_output(InfoPedido)


# --- ESLABÓN 2: APIs Simuladas (Bases de Datos de la Empresa) ---
def consultar_api_logistica(id_pedido: str) -> Dict[str, Any]:
    print(
        f"   [API Logística] Obteniendo datos del envío y cliente para {id_pedido}..."
    )
    if id_pedido == "ORD-5542":
        return {
            "nombre_cliente": "Carlos Mendoza",
            "estado_real": "Retenido en aduanas. Faltan 4 días para liberación.",
        }
    return {
        "nombre_cliente": "Cliente Desconocido",
        "estado_real": "Pedido no encontrado.",
    }


def obtener_perfil_agente(id_agente: str) -> Dict[str, str]:
    print(f"   [API RRHH] Extrayendo firma institucional del agente {id_agente}...")
    # Simulamos que buscamos el ID del agente que está usando el sistema
    if id_agente == "AG-007":
        return {
            "nombre_agente": "Laura Gómez",
            "cargo_agente": "Especialista de Retención de Clientes",
            "empresa_agente": "TechLogistics Global",
        }
    return {
        "nombre_agente": "Soporte",
        "cargo_agente": "Atención al Cliente",
        "empresa_agente": "Nuestra Empresa",
    }


# --- ESLABÓN 3: Política de Resolución ---
resolucion_prompt = ChatPromptTemplate.from_template(
    "Eres el gerente de soporte. El cliente tiene frustración: {emocion}.\n"
    "El pedido indica: {estado_real}.\n"
    "Define la instrucción de cómo el agente debe responder. Sé muy breve."
)

# --- ESLABÓN 4: Redactor del Correo Final ---
redactor_prompt = ChatPromptTemplate.from_template(
    "Escribe un correo formal y empático de máximo 3 líneas dirigido a {nombre_cliente}.\n"
    "Sigue esta instrucción de resolución: {resolucion}\n\n"
    "Firma el correo exactamente con estos datos, sin agregar nada más:\n"
    "{nombre_agente}\n{cargo_agente}\n{empresa_agente}"
)

# ==========================================
# CONSTRUCCIÓN DE LA CADENA CON MEMORIA DE ESTADO (assign)
# ==========================================
workflow = (
    # Entrada inicial: {"correo": "...", "id_agente": "AG-007"}
    # 1. Extraemos info y la SUMAMOS al diccionario bajo la llave 'extraccion'
    RunnablePassthrough.assign(extraccion=extractor)
    # 2. Llamamos a ambas APIs simultáneamente y SUMAMOS los resultados
    | RunnablePassthrough.assign(
        datos_pedido=lambda x: consultar_api_logistica(x["extraccion"].id_pedido),
        perfil_agente=lambda x: obtener_perfil_agente(x["id_agente"]),
    )
    # 3. Preparamos variables para el gerente, ejecutamos y SUMAMOS la 'resolucion'
    | RunnablePassthrough.assign(
        resolucion=(
            lambda x: {
                "emocion": x["extraccion"].emocion,
                "estado_real": x["datos_pedido"]["estado_real"],
            }
        )
        | resolucion_prompt
        | modelo
        | StrOutputParser()
    )
    # 4. Aplanamos todo el diccionario acumulado para inyectarlo al prompt final
    | (
        lambda x: {
            "nombre_cliente": x["datos_pedido"]["nombre_cliente"],
            "resolucion": x["resolucion"],
            "nombre_agente": x["perfil_agente"]["nombre_agente"],
            "cargo_agente": x["perfil_agente"]["cargo_agente"],
            "empresa_agente": x["perfil_agente"]["empresa_agente"],
        }
    )
    | redactor_prompt
    | modelo
    | StrOutputParser()
)

# ==========================================
# EJECUCIÓN
# ==========================================
correo_cliente = "¡Exijo respuestas! Mi pedido ORD-5542 debió llegar ayer. Es un regalo de cumpleaños y estoy furioso."
id_del_agente_en_turno = "AG-007"

print("Iniciando Workflow de Soporte...\n")
respuesta_final = workflow.invoke(
    {"correo": correo_cliente, "id_agente": id_del_agente_en_turno}
)

print("\n--- Respuesta Final Generada ---")
print(respuesta_final)
