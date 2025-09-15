#!/usr/bin/env python3
"""
PostgreSQL Storage Module
Manages the database of videos to track which videos have been downloaded
and processed by which version of the processor system.
"""

import os
import psycopg2
import asyncpg
from psycopg2 import sql
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PostgresStorage:

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """Initialize PostgreSQL connection"""
        # Use environment variables or defaults
        self.host = host or os.environ.get("POSTGRES_HOST", "localhost")
        self.port = port or int(os.environ.get("POSTGRES_PORT", "5432"))
        self.database = database or os.environ.get("POSTGRES_DB", "content_mill")
        self.user = user or os.environ.get("POSTGRES_USER", "content_mill")
        self.password = password or os.environ.get(
            "POSTGRES_PASSWORD", "content_mill_password"
        )

        self.connection = None
        self._connect()
        self.fuck_you_joe()

    def fuck_you_joe(self):
        try:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            scripts_dir = os.path.join(current_dir, "database_setup_scripts")

            # Get all .sql files in the scripts directory and sort them
            sql_files = sorted(glob.glob(os.path.join(scripts_dir, "*.sql")))

            if not sql_files:
                logger.warning(f"No SQL files found in {scripts_dir}")
                return

            with self.connection.cursor() as cursor:
                for sql_file in sql_files:
                    logger.info(f"Executing SQL script: {os.path.basename(sql_file)}")

                    # Read the SQL file content
                    with open(sql_file, "r") as f:
                        sql_content = f.read()

                    # Execute the SQL content
                    cursor.execute(sql_content)

            logger.info("All database setup scripts executed successfully")

        except psycopg2.Error as e:
            logger.error(f"Failed to create/update table: {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading or executing SQL scripts: {e}")
            raise

    def reset_db(self):
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    """
                            DROP TABLE IF EXISTS rooms
                            """
                )
        except:
            print("tuff")

    def _connect(self):
        """Establish connection to PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
            self.connection.autocommit = True
            logger.info(
                f"Connected to PostgreSQL at {self.host}:{self.port}/{self.database}"
            )
        except psycopg2.Error as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def _close(self):
        if self.connection:
            self.connection.close()
            logger.info("Closed PostgreSQL connection")

    def __enter__(self):
        if self.connection is None or self.connection.closed:
            self._connect()  # only auto connect if its called with a context manager - so it auto closes <3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()

    def get_dsn(self) -> str:
        """Get PostgreSQL DSN string for asyncpg connections"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    async def get_async_connection(self) -> asyncpg.Connection:
        """Get an async PostgreSQL connection using asyncpg"""
        return await asyncpg.connect(self.get_dsn())
