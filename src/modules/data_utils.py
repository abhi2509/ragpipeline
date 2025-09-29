import pandas as pd
import pandera as pa
import logging

class DataUtils:
    @staticmethod
    def validate_schema(df, schema=None):
        if schema:
            try:
                schema.validate(df)
                logging.info("Data schema validated.")
            except pa.errors.SchemaError as e:
                logging.error(f"Schema validation failed: {e}")
                raise
        else:
            logging.info("No schema provided for validation.")
        return df

    @staticmethod
    def clean_data(df):
        df = df.drop_duplicates()
        df = df.fillna('N/A')
        logging.info(f"Cleaned data: {df.shape[0]} rows.")
        return df
