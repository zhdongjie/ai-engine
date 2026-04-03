# scripts/init_knowledge_db.py
from ai_engine.infra.db.pgsql import db_manager
from ai_engine.knowledge.initializer import run_init


if __name__ == "__main__":
    db_manager.init_db()
    try:
        run_init()
    finally:
        db_manager.close_db()