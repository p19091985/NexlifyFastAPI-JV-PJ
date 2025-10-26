import logging
from pathlib import Path
from sqlalchemy import create_engine, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, OperationalError

import config
from .security import load_key, decrypt_message

project_root = Path(__file__).parent.parent.resolve()
CONFIG_PATH = project_root / "banco.ini"
SCHEMA_PATH = project_root / "persistencia/sql_schema_SQLLite.sql"

def _set_sqlite_pragma(dbapi_connection, connection_record):
    """Executa o PRAGMA para ativar o suporte a chaves estrangeiras no SQLite."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

class DatabaseManager:
    _engine = None

    @classmethod
    def _parse_active_config(cls):
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"Arquivo de configuração '{CONFIG_PATH}' não encontrado.")
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        active_config = {}
        for line in lines:
            clean_line = line.strip()
            if not clean_line or (clean_line.startswith('#') or clean_line.startswith(';')) or clean_line.startswith(
                    '['):
                continue
            if '=' in clean_line:
                key, value = clean_line.split('=', 1)
                active_config[key.strip()] = value.strip()
        if not active_config or 'type' not in active_config:
            raise ValueError(
                "Nenhuma configuração de banco de dados ativa (descomentada) foi encontrada no 'banco.ini'.")
        return active_config

    @classmethod
    def get_engine(cls):
        if not config.DATABASE_ENABLED:
            logging.warning("Acesso ao banco de dados está desativado em config.py. Nenhuma engine será criada.")
            return None
        if cls._engine is None:
            try:
                db_config = cls._parse_active_config()
                key = load_key()
            except (FileNotFoundError, ValueError, RuntimeError) as e:
                logging.critical(f"Erro ao ler configuração do banco: {e}")
                raise
            except Exception as e:
                logging.critical(f"Falha CRÍTICA ao carregar a chave de segurança: {e}")
                raise RuntimeError("Não foi possível carregar a chave 'secret.key'.") from e

            db_type = db_config.get('type', 'sqlite').lower()
            connection_url = None
            engine_options = {'echo': False}
            logging.info(f"Configuração ativa detectada: '{db_type}'")
            try:
                if db_type == 'sqlite':
                    db_path = project_root / db_config.get('path', 'sistema.db')
                    connection_url = f"sqlite:///{db_path}"
                    engine_options['connect_args'] = {'timeout': 15}

                    engine = create_engine(connection_url, **engine_options)

                    event.listen(engine, "connect", _set_sqlite_pragma)
                    cls._engine = engine
                else:
                    user = decrypt_message(db_config['user'], key)
                    password = decrypt_message(db_config['password'], key)
                    host = db_config['host']
                    dbname = db_config['dbname']
                    port = db_config.get('port')

                    if db_type == 'postgresql':
                        connection_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
                    elif db_type == 'mysql':
                        connection_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{dbname}"
                    elif db_type == 'sqlserver':
                        connection_url = f"mssql+pymssql://{user}:{password}@{host}:{port}/{dbname}"
                    elif db_type == 'mariadb':
                        connection_url = f"mariadb+mariadbconnector://{user}:{password}@{host}:{port}/{dbname}"
                    elif db_type == 'oracle':
                        dsn = f"{host}:{port}/{dbname}"
                        connection_url = f"oracle+oracledb://{user}:{password}@{dsn}"
                    elif db_type == 'firebird':
                        connection_url = f"firebird+fdb://{user}:{password}@{host}:{port}/{dbname}"
                    else:
                        raise ValueError(f"Tipo de banco de dados não suportado: '{db_type}'")

                    cls._engine = create_engine(connection_url, **engine_options)

                with cls._engine.connect() as connection:
                    logging.info(f"Conexão com '{db_type}' estabelecida com sucesso.")
            except (OperationalError, SQLAlchemyError) as e:
                logging.error(
                    f"Erro ao conectar ao banco '{db_type}'. Verifique as credenciais, rede e status do servidor.")
                raise ConnectionError(f"Não foi possível conectar ao banco '{db_type}'.") from e
            except KeyError as e:
                logging.error(f"Parâmetro de configuração faltando no banco.ini para '{db_type}': {e}")
                raise KeyError(f"Parâmetro '{e}' faltando no 'banco.ini' para a conexão '{db_type}'.") from e
            except Exception as e:
                logging.error(f"Erro inesperado durante a configuração do banco: {e}")
                raise
        return cls._engine

    @classmethod
    def initialize_database(cls):
        engine = cls.get_engine()
        if not engine:
            logging.error("Não foi possível inicializar o banco: engine não disponível.")
            return
        if engine.url.drivername != 'sqlite':
            logging.info("Inicialização de schema pulada para banco não-SQLite.")
            return
        db_path = Path(engine.url.database)
        if db_path.exists() and db_path.stat().st_size > 0:
            logging.info("Banco de dados SQLite já parece estar inicializado.")
            return
        if not SCHEMA_PATH.is_file():
            raise FileNotFoundError(f"Arquivo de schema não encontrado em {SCHEMA_PATH}")
        logging.info(f"Inicializando banco de dados SQLite a partir de '{SCHEMA_PATH}'...")
        try:
            schema_sql = SCHEMA_PATH.read_text(encoding='utf-8')
            with engine.connect() as conn:
                for command in schema_sql.split(';'):
                    if command.strip():
                        conn.execute(text(command))
                conn.commit()
            logging.info("Banco de dados SQLite inicializado com sucesso.")
        except Exception as e:
            logging.error(f"Erro ao executar o script de inicialização do SQLite: {e}")
            raise