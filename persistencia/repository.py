import pandas as pd
from sqlalchemy import text, exc
import logging
import config
from .database import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GenericRepository:
    """
    Classe genérica para interagir com o banco de dados.
    Verifica se o banco está habilitado antes de cada operação.
    """

    @staticmethod
    def get_engine():
        """Retorna a instância do motor do DatabaseManager."""
        return DatabaseManager.get_engine()

    @staticmethod
    def execute_query_to_dataframe(query: str, params: dict = None):
        """Executa uma query e retorna um DataFrame com colunas minúsculas."""
        if not config.DATABASE_ENABLED:
            logging.warning("Banco de dados desabilitado. A query não será executada.")
            return pd.DataFrame()

        engine = GenericRepository.get_engine()
        if not engine:
            logging.error("Acesso ao banco falhou: engine não disponível.")
            return pd.DataFrame()

        try:
            with engine.connect() as connection:
                df = pd.read_sql_query(text(query), connection, params=params)
                                                     
                df.columns = [str(col).lower() for col in df.columns]
                return df
        except exc.SQLAlchemyError as e:
            logging.error(f"Erro ao executar a query: {query}\nErro: {e}")
            raise

    @staticmethod
    def write_dataframe_to_table(df: pd.DataFrame, table_name: str):
        """Escreve um DataFrame em uma tabela (espera nome da tabela minúsculo)."""
        if not config.DATABASE_ENABLED:
            logging.warning(f"Banco de dados desabilitado. Nenhum dado será escrito em '{table_name}'.")
            return

        engine = GenericRepository.get_engine()
        if not engine:
            logging.error(f"Escrita em '{table_name}' falhou: engine não disponível.")
            return

        try:
                                                                                
            df_to_write = df.copy()
            df_to_write.columns = [str(col).lower() for col in df_to_write.columns]
                                                                      
            df_to_write.to_sql(table_name, con=engine, if_exists='append', index=False)
            logging.info(f"{len(df)} registros inseridos com sucesso na tabela '{table_name}'.")
        except exc.SQLAlchemyError as e:
                                                   
            logging.error(f"Erro ao escrever na tabela '{table_name}'. Colunas do DF: {list(df.columns)}. Erro: {e}")
            raise

    @staticmethod
    def update_table(table_name: str, update_values: dict, where_conditions: dict):
        """Atualiza registros em uma tabela (espera nome da tabela e chaves minúsculas)."""
        if not config.DATABASE_ENABLED:
            logging.warning(f"Banco de dados desabilitado. Nenhum dado será atualizado em '{table_name}'.")
            return

        engine = GenericRepository.get_engine()
        if not engine:
            logging.error(f"Update em '{table_name}' falhou: engine não disponível.")
            return

        update_values_lower = {k.lower(): v for k, v in update_values.items()}
        where_conditions_lower = {k.lower(): v for k, v in where_conditions.items()}

        set_clause = ", ".join([f"{key} = :{key}_val" for key in update_values_lower.keys()])
        where_clause = " AND ".join([f"{key} = :wh_{key}" for key in where_conditions_lower.keys()])

        params = {f'{k}_val': v for k, v in update_values_lower.items()}
        params.update({f'wh_{k}': v for k, v in where_conditions_lower.items()})

        query = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause}"

        try:
            with engine.connect() as connection:
                with connection.begin():
                    connection.execute(text(query), params)
            logging.info(f"Tabela '{table_name}' atualizada com sucesso.")
        except exc.SQLAlchemyError as e:
            logging.error(f"Erro ao atualizar a tabela '{table_name}': {e}")
            raise

    @staticmethod
    def delete_from_table(table_name: str, where_conditions: dict):
        """Deleta registros de uma tabela (espera nome da tabela e chaves minúsculas)."""
        if not config.DATABASE_ENABLED:
            logging.warning(f"Banco de dados desabilitado. Nenhum dado será deletado de '{table_name}'.")
            return

        engine = GenericRepository.get_engine()
        if not engine:
            logging.error(f"Delete em '{table_name}' falhou: engine não disponível.")
            return

        where_conditions_lower = {k.lower(): v for k, v in where_conditions.items()}

        where_clause = " AND ".join([f"{key} = :{key}" for key in where_conditions_lower.keys()])
        params = where_conditions_lower

        query = f"DELETE FROM {table_name} WHERE {where_clause}"

        try:
            with engine.connect() as connection:
                with connection.begin():
                    connection.execute(text(query), params)
            logging.info(f"Registros da tabela '{table_name}' deletados com sucesso.")
        except exc.SQLAlchemyError as e:
            logging.error(f"Erro ao deletar da tabela '{table_name}': {e}")
            raise

    @staticmethod
    def read_table_to_dataframe(table_name: str, columns: list = None, where_conditions: dict = None):
        """Lê dados de uma tabela (espera nome da tabela minúsculo) e retorna DataFrame."""
        if not config.DATABASE_ENABLED:
            return pd.DataFrame()

        cols_str = "*"
        if columns:
             cols_lower = [col.lower() for col in columns]
             cols_str = ", ".join(cols_lower)

        query = f"SELECT {cols_str} FROM {table_name}"

        params_lower = None
        if where_conditions:
                                                                              
            where_conditions_lower = {k.lower(): v for k, v in where_conditions.items()}
            conditions = " AND ".join([f"{key} = :{key}" for key in where_conditions_lower.keys()])
            query += f" WHERE {conditions}"
            params_lower = where_conditions_lower                                         

        return GenericRepository.execute_query_to_dataframe(query, params=params_lower)