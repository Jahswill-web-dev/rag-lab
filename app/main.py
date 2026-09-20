import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.db.session import create_engine, create_session_factory
from app.services.answers import OpenAIAnswerService
from app.services.embeddings import OpenAIEmbeddingService

# Psycopg's async driver requires a selector loop on Windows. Linux deployments
# keep their default event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
        engine = create_engine(resolved_settings)
        application.state.engine = engine
        application.state.session_factory = create_session_factory(engine)
        embedding_service = OpenAIEmbeddingService.from_settings(resolved_settings)
        application.state.embedding_service = embedding_service
        answer_service = OpenAIAnswerService.from_settings(resolved_settings)
        application.state.answer_service = answer_service

        try:
            yield
        finally:
            await answer_service.close()
            await embedding_service.close()
            await engine.dispose()

    app = FastAPI(
        title=resolved_settings.app_name,
        debug=resolved_settings.debug,
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    app.include_router(ask_router)
    return app


app = create_app()
