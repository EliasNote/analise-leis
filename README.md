# Juridico AI — Análise de Leis para ME/EPP

Aplicação de consultoria jurídica assistida por IA, focada em **Direito Empresarial para Micro e Pequenas Empresas (ME/EPP)**. O sistema combina **RAG (Retrieval Augmented Generation)** com **agentes especializados** (tributário, trabalhista, societário) e uma interface **Streamlit** para chat e ingestão de leis em base vetorial **Qdrant**.

## Visão geral

- **Chat jurídico** com classificação automática da intenção.
- **Agentes especialistas** com prompts orientados à legislação aplicável.
- **RAG com Qdrant** para responder com base em textos legais indexados.
- **Ingestão de leis via URL** (Planato/HTML) com divisão por artigos.
- **Autenticação simples** via usuário/senha configuráveis por variável de ambiente.

## Arquitetura (alto nível)

1. **UI (Streamlit)**: autenticação, chat e gestão de leis.
2. **Router**: classifica a pergunta do usuário em `tributario`, `trabalhista`, `societario`, `conversational` ou `out_of_scope`.
3. **Agentes especializados (PydanticAI)**: cada domínio possui prompt e ferramentas próprias.
4. **RAG (LlamaIndex + Qdrant)**: busca trechos relevantes na base vetorial.
5. **Modelos (Bedrock)**: LLM para geração e modelo de embeddings.

## Componentes principais

- **app.py**: interface Streamlit, login, chat e ingestão.
- **main.py**: orquestração do fluxo com LangGraph.
- **Agents.py**: agentes PydanticAI e ferramentas de busca.
- **Prompts.py**: templates de prompts e regras de resposta.
- **ingestion.py**: ingestão e indexação de leis no Qdrant.
- **utils.py**: extração de HTML e fatiamento por artigos.
- **LLM.py**: configuração dos modelos Bedrock.

## Fluxo de resposta (chat)

1. Usuário envia uma pergunta no chat.
2. O **router** classifica o tema.
3. O agente especializado executa **busca RAG** no Qdrant.
4. O agente gera a resposta com referências legais.

## Fluxo de ingestão de leis

1. Usuário informa URLs de leis (uma por linha).
2. O sistema baixa o HTML, limpa ruídos e identifica artigos.
3. Os artigos são **fatiados** e **indexados** em lotes no Qdrant.
4. A tela exibe progresso e logs detalhados.

## Stack e dependências

- **Python**: recomendado **3.12+**.
- **Streamlit** (UI)
- **LangGraph + PydanticAI** (agentes)
- **LlamaIndex** (RAG)
- **Qdrant** (vetores)
- **AWS Bedrock** (LLM/Embeddings)

## Variáveis de ambiente

Crie um arquivo .env com as chaves da AWS e parâmetros da aplicação.

Obrigatórias:

- `QDRANT_URL`: URL do Qdrant (ex.: http://localhost:6333)

Recomendadas (login):

- `APP_USER`
- `APP_PASSWORD`

AWS Bedrock (necessário para produção):

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`

## Como executar (local)

1. Configure o .env.
2. Instale dependências.
3. Inicie o Streamlit.

Observação: o projeto usa `uv` no Docker, mas pode ser executado localmente com seu gerenciador padrão.

## Como executar (Docker)

O docker-compose já sobe a aplicação e o Qdrant:

- **Qdrant**: porta 6333
- **Streamlit**: porta 8501

O serviço `app` lê o .env local para credenciais da AWS.

## Uso da aplicação

- **Login**: usa `APP_USER` e `APP_PASSWORD`.
- **Chat**: faça perguntas sobre tributário, trabalhista ou societário.
- **Gestão de Leis**: adicione URLs de leis e veja as fontes indexadas.

## Estrutura do projeto

- app.py — UI e fluxo principal
- main.py — grafo de orquestração
- Agents.py — agentes e ferramentas
- Prompts.py — prompts e regras
- ingestion.py — pipeline de ingestão
- utils.py — parsing e fatiamento
- LLM.py — configuração dos modelos

## Observações importantes

- A base vetorial é persistida em `qdrant_data/`.
- O projeto pressupõe acesso ao **AWS Bedrock**.
- A coleção utilizada no Qdrant é `leis_v3`.

## Troubleshooting

- **Erro ao carregar IA**: verifique `QDRANT_URL` e credenciais AWS.
- **Sem resultados no chat**: confirme se há leis indexadas na coleção.
- **Timeouts de ingestão**: reduza o número de URLs por vez.
