import asyncio
import json
import os
import logging

from azure.identity import ManagedIdentityCredential
from microsoft_teams.ai import ChatPrompt, ListMemory
from microsoft_teams.ai.ai_model import AIModel
from microsoft_teams.apps import App, ActivityContext
from microsoft_teams.openai import OpenAICompletionsAIModel
from microsoft_teams.api import CitationAppearance, MessageActivity, MessageActivityInput, MessageSubmitActionInvokeActivity

from config import Config
from my_data_source import MyDataSource
from groundedness_engine import TeamsGroundednessEngine
from groundedness_monitoring import GroundednessMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()
my_data_source = MyDataSource()

# Initialize groundedness engine with monitoring
groundedness_monitor = GroundednessMonitor(log_dir="logs")
groundedness_engine = TeamsGroundednessEngine(monitor=groundedness_monitor)

# Load instructions from file
def load_instructions() -> str:
    """Load instructions from instructions.txt file"""
    try:
        with open(os.path.join(os.path.dirname(__file__), "instructions.txt"), "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are a helpful assistant."

INSTRUCTIONS = load_instructions()

def create_token_factory():
    def get_token(scopes, tenant_id=None):
        credential = ManagedIdentityCredential(client_id=config.APP_ID)
        if isinstance(scopes, str):
            scopes_list = [scopes]
        else:
            scopes_list = scopes
        token = credential.get_token(*scopes_list)
        return token.token
    return get_token

app = App(
    token=create_token_factory() if config.APP_TYPE == "UserAssignedMsi" else None
)

# Initialize AI model for chat
model = OpenAICompletionsAIModel(
    api_key=config.OPENAI_API_KEY,
    org_id=config.OPENAI_ORG_ID if hasattr(config, 'OPENAI_ORG_ID') else None,
    default_model="gpt-35-turbo"
)

conversation_store: dict[str, ListMemory] = {}

def get_or_create_conversation_memory(conversation_id: str) -> ListMemory:
    """Get or create conversation memory for a specific conversation"""
    if conversation_id not in conversation_store:
        conversation_store[conversation_id] = ListMemory()
    return conversation_store[conversation_id]

async def handle_stateful_conversation(model: AIModel, ctx: ActivityContext[MessageActivity]) -> None:
    """Example of stateful conversation handler that maintains conversation history"""
    # Retrieve existing conversation memory or initialize new one
    memory = get_or_create_conversation_memory(ctx.activity.conversation.id)

    # Get existing messages for logging
    existing_messages = await memory.get_all()
    print(f"Existing messages before sending to prompt: {len(existing_messages)} messages")

    input = ctx.activity.strip_mentions_text().text
    data_context = my_data_source.render_data(input)

    # Create ChatPrompt with conversation-specific memory
    chat_prompt = ChatPrompt(model)

    try:
        chat_result = await chat_prompt.send(
            input=input,
            memory=memory,
            instructions=f"{INSTRUCTIONS}\n\nAdditional Context:\n${data_context.output}"
        )
    except Exception as e:
        print(f"Error sending chat prompt: {e}")
        await ctx.send(MessageActivityInput(text="An error occurred while processing your request."))
        return

    result = None
    try:
        # Attempt to parse the response as JSON
        result = json.loads(chat_result.response.content)
    except json.JSONDecodeError as error:
        print(f"Error decoding JSON: {error}")
        await ctx.send(MessageActivityInput(text=chat_result.response.content).add_ai_generated().add_feedback())
        return

    
    citations = []
    position = 1
    content = ""
    extracted_answer = ""  # For groundedness testing
    evidence_list = []  # For groundedness testing

    if result and result.get("results") and len(result["results"]) > 0:
        for content_item in result["results"]:
            
            if content_item.get("citationTitle") and len(content_item["citationTitle"]) > 0:
                answer_text = content_item['answer']
                extracted_answer += answer_text  # Accumulate for testing
                content += f"{answer_text}[{position}]<br>"
                
                # Extract evidence from citation
                evidence_list.append({
                    "id": str(position),
                    "text": content_item.get("citationContent", ""),
                    "source": content_item.get("citationTitle", "")
                })

                citations.append(
                    {
                        "id": position,
                        "title": content_item.get("citationTitle", ""),
                        "abstract": content_item.get("citationContent", "")[:160]
                    }
                )
                position += 1
            else:
                answer_text = content_item['answer']
                extracted_answer += answer_text
                content += f"{answer_text}<br>"
    
    # Test groundedness of the extracted answer
    if extracted_answer:
        try:
            verdict = await groundedness_engine.test_response(
                query=input,
                answer=extracted_answer,
                evidence=evidence_list,
                conversation_id=ctx.activity.conversation.id
            )
            
            # Add verdict to response
            verdict_text = verdict.get_verdict_text()
            score_display = f"[Confidence: {verdict.confidence_level.upper()} {verdict.score:.0%}]"
            
            # Prepend verdict to content
            content = f"<strong>{verdict_text}</strong><br>{score_display}<br><br>{content}"
            
            logger.info(f"Groundedness verdict for '{input[:30]}...': {verdict.confidence_level} ({verdict.score:.1%})")
            
        except Exception as e:
            logger.error(f"Error testing groundedness: {e}")
            # Continue with message even if testing fails
    
    message_activity = MessageActivityInput(text=content).add_ai_generated().add_feedback()
    for citation in citations:
        message_activity.add_citation(
            citation["id"],
            CitationAppearance(name=citation["title"], abstract=citation["abstract"])
        )

    await ctx.send(message_activity)

@app.on_message
async def handle_message(ctx: ActivityContext[MessageActivity]):
    """Handle messages using stateful conversation"""
    await handle_stateful_conversation(model, ctx)

@app.on_message_submit_feedback
async def handle_message_feedback(ctx: ActivityContext[MessageSubmitActionInvokeActivity]):
    """Handle feedback submission events"""
    activity = ctx.activity

    print(f"your feedback is {activity.value.action_value}")

if __name__ == "__main__":
    asyncio.run(app.start())