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