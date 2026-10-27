"""Run the real ADK agent once per question; never pass reference answers to it."""

import asyncio
from contextlib import AsyncExitStack, aclosing

from eval.rag.promptfoo.capture import capture


async def run_question(question: str) -> dict:
    # Lazy imports allow offline contract tests without cloud configuration.
    from google.adk.agents.run_config import RunConfig
    from google.adk.models import Gemini
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from app.adapters.llm.embeddings_service import EmbeddingsClient
    from app.adapters.persistence.database import close_db, init_db
    from app.components.agents.root.agent import assistant_agent, root_agent
    from app.config.settings import settings

    async with AsyncExitStack() as stack:
        stack.push_async_callback(close_db)
        stack.push_async_callback(EmbeddingsClient().close)
        await init_db()
        model = Gemini(
            model=settings.MODEL,
            client_kwargs={
                "vertexai": True,
                "project": settings.GOOGLE_CLOUD_PROJECT,
                "location": settings.GOOGLE_CLOUD_LOCATION,
                "http_options": types.HttpOptions(
                    timeout=30000,
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            },
        )
        stack.callback(model.api_client.close)
        stack.push_async_callback(model.api_client.aio.aclose)
        previous_model = assistant_agent.model
        assistant_agent.model = model
        stack.callback(setattr, assistant_agent, "model", previous_model)
        runner = InMemoryRunner(node=root_agent, app_name="rag_promptfoo")
        stack.push_async_callback(runner.close)
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id="promptfoo"
        )
        events = []
        async with aclosing(
            runner.run_async(
                user_id="promptfoo",
                session_id=session.id,
                new_message=types.Content(
                    role="user", parts=[types.Part(text=question)]
                ),
                run_config=RunConfig(max_llm_calls=10),
            )
        ) as stream:
            async for event in stream:
                events.append(event)
        return capture(events)


async def call_api(prompt: str, options: dict, context: dict) -> dict:
    """Promptfoo Python provider contract; context/reference is deliberately unused."""
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "A nonempty question is required"}
    try:
        async with asyncio.timeout(180):
            return {"output": await run_question(prompt)}
    except Exception as error:
        # Connection strings, cloud identifiers and provider errors stay private.
        return {"error": f"ADK evaluation failed ({type(error).__name__})"}
