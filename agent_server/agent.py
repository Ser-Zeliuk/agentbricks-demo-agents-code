from typing import AsyncGenerator, Optional

import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks, DatabricksMCPServer, DatabricksMultiServerMCPClient
from langchain.agents import create_agent
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    to_chat_completions_input,
)

from agent_server.utils import (
    get_databricks_host_from_env,
    get_user_workspace_client,
    process_agent_astream_events,
)

mlflow.langchain.autolog()
sp_workspace_client = WorkspaceClient()

SYSTEM_PROMPT = """
You are a research-planning agent for biomedical questions.

User questions will look like: “Does drug X help for disease Y?”
Your job is NOT to give medical advice. Do not recommend treatment for any individual.
You must produce an evidence-grounded research overview and a NEW experiment proposal.
You must provide direct references to all the data (research information and experiment best practices). Use file names, from which the data was extracted.

Core workflow (must follow in order):
1) Retrieve evidence:
   - Use the vector-search tool “docs-search” to find existing research and any internal guidelines.
   - Retrieve at least 5 relevant sources/snippets when possible.
   - If retrieval returns insufficient or off-topic info, say so and proceed cautiously.

2) Summarize existing research:
   - Provide a concise synthesis: what is known, strength of evidence, key outcomes measured, limitations.
   - Clearly separate preclinical vs clinical vs observational evidence if present.
   - Do NOT claim efficacy unless the retrieved evidence supports it, and state uncertainty.

3) Identify knowledge gaps:
   - Call the “identify_knowledge_gaps” function/tool (or equivalent).
   - If the tool isn’t available, infer gaps from the retrieved evidence and label them as “inferred”.

4) Design a brand new experiment:
   - Use the experiment guidelines retrieved from vector-search as methodology constraints.
   - The experiment must be new (not a copy of any retrieved protocol), but consistent with the guidelines.
   - Choose the most appropriate study type (in vitro / in vivo / clinical trial / observational / RWE),
     and justify why.
   - Include: hypothesis, endpoints, inclusion/exclusion (if human), controls, sample size rationale (high-level),
     randomization/blinding (if applicable), procedures, timeline, data analysis plan, risks/ethics.

5) Materials and cost:
   - Call “extract_materials” to list required materials/resources.
   - Call “calculate_costs” to estimate costs, with assumptions and ranges.
   - If tools return incomplete outputs, fill gaps with clearly labeled estimates.

Output requirements:
- Always return a final report with the exact sections:
  A) Research overview
  B) Evidence table (bullets are fine)
  C) Knowledge gaps
  D) Proposed new experiment design
  E) Materials
  F) Cost estimate (with assumptions)
  G) Safety / ethics notes
- Include citations as “Retrieved snippet:” with short quotes or paraphrases and identifiers (doc title/ID if available).
- Never invent citations. Only cite what you retrieved.

Tool-use rules:
- Use tools whenever you need facts from the workspace or guidelines.
- Do not fabricate tool outputs. If a tool fails, say it failed and proceed with best-effort reasoning.
"""



def init_mcp_client(workspace_client: WorkspaceClient) -> DatabricksMultiServerMCPClient:
    host_name = get_databricks_host_from_env()
    return DatabricksMultiServerMCPClient(
        [
            DatabricksMCPServer(
                name="system-ai",
                url=f"{host_name}/api/2.0/mcp/functions/system/ai",
                workspace_client=workspace_client,
                handle_tool_error="Tool call failed. Explain what you were trying to do and continue with best-effort.",

            ),
            DatabricksMCPServer(
                name="agenbrick-demo",
                url=f"{host_name}/api/2.0/mcp/functions/agenbrick_demo/default",
                workspace_client=workspace_client,
                handle_tool_error="Tool call failed. Explain what you were trying to do and continue with best-effort.",

            ),
            DatabricksMCPServer.from_vector_search(catalog="demo_location", # <- your UC catalog 
                                                   schema="demo", # <- your UC schema 
                                                #    index_name="combined_text_data_index", # <- your vector search index name 
                                                   name="docs-search", workspace_client=workspace_client),
            
        ]
    )


async def init_agent(workspace_client: Optional[WorkspaceClient] = None):
    mcp_client = init_mcp_client(workspace_client or sp_workspace_client)
    tools = await mcp_client.get_tools()
    return create_agent(tools=tools, model=ChatDatabricks(endpoint="databricks-claude-sonnet-4-6"))


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    outputs = [
        event.item
        async for event in streaming(request)
        if event.type == "response.output_item.done"
    ]
    return ResponsesAgentResponse(output=outputs)


@stream()
async def streaming(
    request: ResponsesAgentRequest,
) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    # Optionally use the user's workspace client for on-behalf-of authentication
    # user_workspace_client = get_user_workspace_client()
    agent = await init_agent()
    user_msgs = [i.model_dump() for i in request.input]
    chat_msgs = to_chat_completions_input(user_msgs)
    chat_msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_msgs
    messages = {"messages": chat_msgs}

    async for event in process_agent_astream_events(
        agent.astream(input=messages, stream_mode=["updates", "messages"])
    ):
        yield event
