CREATE TABLE IF NOT EXISTS titanic_data (PASSENGERID integer,SURVIVED integer,PCLASS integer,NAME varchar,SEX varchar,AGE integer,
SIBSP double,PARCH double,TICKET varchar,FARE double,CABIN varchar,EMBARKED varchar);

COPY titanic_data FROM '../../../examples/dumps/test_data/titanic.csv' WITH (FORMAT csv);
