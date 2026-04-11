from collections.abc import Iterator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, echo=False, future=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def ensure_runtime_schema() -> None:
    expected_columns = {
        "documents": {
            "document_class": "VARCHAR(32)",
            "document_family": "VARCHAR(32)",
        },
        "versions": {
            "document_class": "VARCHAR(32)",
            "document_family": "VARCHAR(32)",
            "chunking_strategy": "VARCHAR(64)",
            "classification_source": "VARCHAR(32)",
            "classification_confidence": "FLOAT",
            "classification_reason": "TEXT",
        },
        "chunks": {
            "indexed_profile": "VARCHAR(128)",
        },
    }

    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            existing = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, ddl_type in columns.items():
                if column_name in existing:
                    continue
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {ddl_type}")
                )


def get_db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
