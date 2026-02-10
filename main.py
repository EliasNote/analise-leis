import logging
from dataclasses import dataclass, field
from typing import List

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.base import BaseCheckpointSaver
import Agents
from pydantic import BaseModel, Field

from Prompts import juiz_tmpl
from LLM import sonnet_bedrock_model

TRIBUTARIO = "tributario"
TRABALHISTA = "trabalhista"
SOCIETARIO = "societario"
CONVERSATIONAL = "conversational"
OUT_OF_SCOPE = "out_of_scope"

class AvaliacaoJuiz(BaseModel):
    nota: int = Field(description="Nota de 1 a 5 para a resposta")
    justificativa: str = Field(description="Explicação do porquê da nota")
    tem_alucinacao: bool = Field(description="Se a IA inventou fatos fora dos documentos")
    correcao_necessaria: str = Field(description="O que o agente deve mudar se a nota for baixa")

# =======================================================
# 1. ESTADO DO WORKFLOW
# =======================================================
@dataclass
class WorkflowState:
    user_question: str
    chat_history: List[str] = field(default_factory=list)
    classification_profile: str = None
    final_response: str = None

# =======================================================
# 2. HELPER (Auxiliar para atualizar memória)
# =======================================================
def _atualizar_historico(state: WorkflowState, resposta_ai: str) -> List[str]:
    """
    Pega o histórico antigo e adiciona a interação atual.
    O LangGraph salvará essa nova lista no Postgres automaticamente.
    """
    nova_interacao = [
        f"User: {state.user_question}",
        f"AI: {resposta_ai}"
    ]
    return state.chat_history + nova_interacao

# =======================================================
# 2. IMPLEMENTAÇÃO DOS NÓS
# =======================================================
_engine_instance = None

def _preparar_dependencias(state: WorkflowState) -> Agents.LegalDeps:
    historico_str = "\n".join(state.chat_history)
    return Agents.LegalDeps(
        query_engine=_engine_instance,
        historico_conversa=historico_str
    )

async def node_router(state: WorkflowState):
    logging.info("--- ROUTER: Classificando perfil ---")
    deps = _preparar_dependencias(state)
    result = await Agents.router_agent.run(state.user_question, deps=deps)
    
    raw_response = str(result.output).strip().lower()
    profile = raw_response.split()[0].replace(".", "").replace(",", "")
    
    profiles = [TRIBUTARIO, TRABALHISTA, SOCIETARIO, CONVERSATIONAL, OUT_OF_SCOPE]

    if profile not in profiles:
       logging.warning(f"Router retornou classificação inválida: '{profile}'. Redirecionando para OUT_OF_SCOPE.")
       profile = OUT_OF_SCOPE
    
    logging.info(f"--- ROUTER: Perfil definido como '{profile}' ---")
    return {"classification_profile": profile}

async def node_tributario(state: WorkflowState):
    logging.info("--- AGENTE: Simples Nacional, ME/EPP e Pronampe ---")
    deps = _preparar_dependencias(state)
    result = await Agents.tributario_agent.run(state.user_question, deps=deps)
    
    resp = str(result.output)
    # Atualiza histórico e resposta
    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }

async def node_trabalhista(state: WorkflowState):
    logging.info("--- AGENTE: Trabalhista (CLT) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.trabalhista_agent.run(state.user_question, deps=deps)
    
    resp = str(result.output)
    # Atualiza histórico e resposta
    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }

async def node_societario(state: WorkflowState):
    logging.info("--- AGENTE: Societario (Burocracia / Lei 14.195) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.societario_agent.run(state.user_question, deps=deps)
    
    resp = str(result.output)
    # Atualiza histórico e resposta
    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }

async def node_conversational(state: WorkflowState):
    logging.info("--- AGENTE: Conversational (Papo Social) ---")
    deps = _preparar_dependencias(state)
    result = await Agents.conversational_agent.run(state.user_question, deps=deps)
    
    resp = str(result.output)
    # Atualiza histórico e resposta
    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }

async def node_out_of_scope(state: WorkflowState):
    resp = (
        "Desculpe, não posso ajudar com esse assunto.\n\n"
        "Minha base de conhecimento é restrita e especializada apenas em:\n"
        "1. **Tributário:** Simples Nacional e Pronampe (LC 123/2006 e Lei 13.999);\n"
        "2. **Trabalhista:** Regras da CLT e contratações;\n"
        "3. **Societário:** Abertura de empresas e Lei do Ambiente de Negócios (Lei 14.195).\n\n"
        "Para outros temas jurídicos (como Criminal, Família, ou Falências) ou assuntos gerais, não tenho informações disponíveis."
    )

    return {
        "final_response": resp,
        "chat_history": _atualizar_historico(state, resp)
    }
    

async def no_juiz(state: AgentState):
    # O Juiz recebe a pergunta, o que foi buscado no Qdrant e a resposta do agente
    contexto = state["contexto_recuperado"]
    pergunta = state["user_question"]
    resposta_candidata = state["final_response"]

    # Usamos o modelo mais forte (Sonnet) para julgar
    juiz = sonnet_bedrock_model.with_structured_output(AvaliacaoJuiz)
    veredito = await juiz.ainvoke(juiz_tmpl)

    return {"avaliacao": veredito}

# =======================================================
# 3. LÓGICA E CRIAÇÃO
# =======================================================
def check_profile_logic(state: WorkflowState):
    if state.classification_profile == TRIBUTARIO:
        return TRIBUTARIO
    elif state.classification_profile == TRABALHISTA:
        return TRABALHISTA
    elif state.classification_profile == SOCIETARIO:
        return SOCIETARIO
    elif state.classification_profile == CONVERSATIONAL:
        return CONVERSATIONAL
    else:
        return OUT_OF_SCOPE

def create_workflow(query_engine, checkpointer: BaseCheckpointSaver = None):
    global _engine_instance
    _engine_instance = query_engine

    NODE_ROUTER = "node_router"
    NODE_TRIBUTARIO = f"node_{TRIBUTARIO}"
    NODE_TRABALHISTA = f"node_{TRABALHISTA}"
    NODE_SOCIETARIO = f"node_{SOCIETARIO}"
    NODE_CONVERSATIONAL = f"node_{CONVERSATIONAL}"
    NODE_OUT_OF_SCOPE = f"node_{OUT_OF_SCOPE}"
    
    workflow = StateGraph(WorkflowState)
    workflow.add_node(NODE_ROUTER, node_router)
    workflow.add_node(NODE_TRIBUTARIO, node_tributario)
    workflow.add_node(NODE_TRABALHISTA, node_trabalhista)
    workflow.add_node(NODE_SOCIETARIO, node_societario)
    workflow.add_node(NODE_CONVERSATIONAL, node_conversational)
    workflow.add_node(NODE_OUT_OF_SCOPE, node_out_of_scope)
    
    workflow.add_edge(START, NODE_ROUTER)

    mapa_decisao = {
        TRIBUTARIO: NODE_TRIBUTARIO,
        TRABALHISTA: NODE_TRABALHISTA,
        SOCIETARIO: NODE_SOCIETARIO,
        CONVERSATIONAL: NODE_CONVERSATIONAL,
        OUT_OF_SCOPE: NODE_OUT_OF_SCOPE
    }
    
    workflow.add_conditional_edges(
        NODE_ROUTER,
        check_profile_logic,
        mapa_decisao
    )
    
    workflow.add_edge(NODE_TRIBUTARIO, END)
    workflow.add_edge(NODE_TRABALHISTA, END)
    workflow.add_edge(NODE_SOCIETARIO, END)
    workflow.add_edge(NODE_CONVERSATIONAL, END)
    workflow.add_edge(NODE_OUT_OF_SCOPE, END)
    
    return workflow.compile(checkpointer=checkpointer)