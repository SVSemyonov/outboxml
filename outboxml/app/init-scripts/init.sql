CREATE TABLE IF NOT EXISTS titanic_data (
    "PASSENGERID" integer,
    "SURVIVED" integer,
    "PCLASS" integer,
    "NAME" varchar,
    "SEX" varchar,
    "AGE" double precision,
    "SIBSP" double precision,
    "PARCH" double precision,
    "TICKET" varchar,
    "FARE" double precision,
    "CABIN" varchar,
    "EMBARKED" varchar
);

-- Указываем путь, который мы задали в volumes (внутри контейнера)
COPY titanic_data
FROM '/docker-entrypoint-initdb.d/titanic.csv'
WITH (FORMAT csv, HEADER true, DELIMITER ',');

-- Уведомление для outboxml.automl_utils.check_postgre_transaction (канал table_changes)
CREATE OR REPLACE FUNCTION notify_table_changes()
RETURNS trigger AS $$
BEGIN
    PERFORM pg_notify(
        'table_changes',
        format('DML event: %s on table %s', TG_OP, TG_TABLE_NAME)
    );
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER titanic_data_changes
AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON titanic_data
FOR EACH STATEMENT
EXECUTE FUNCTION notify_table_changes();
