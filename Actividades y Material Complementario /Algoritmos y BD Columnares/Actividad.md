# Ejercicio: PostgreSQL vs DuckDB — EXPLAIN en acción

**IIC2440 — Procesamiento de Datos Masivos — Clase 03**

En este ejercicio van a ejecutar las mismas consultas en dos motores de base de datos con arquitecturas completamente diferentes:
- **PostgreSQL**: row-store, ejecución tuple-at-a-time
- **DuckDB**: column-store, ejecución vectorizada

El objetivo es **observar y entender** cómo cada motor elige algoritmos distintos y por qué.

---

## 0. Setup

### Instalar PostgreSQL

**Mac (Homebrew)**:
```bash
brew install postgresql@17
brew services start postgresql@17
```

**Mac (Postgres.app)**:
1. Descargar desde https://postgresapp.com/ e instalar arrastrando a Aplicaciones.
2. Abrir Postgres.app y hacer click en "Initialize" para crear el servidor.
3. Para tener `psql` disponible en el terminal, agregar al PATH:
```bash
export PATH="/Applications/Postgres.app/Contents/Versions/latest/bin:$PATH"
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
```

**Windows**: Descargar el instalador desde https://www.postgresql.org/download/windows/ y seguir el wizard. Asegurarse de que el servicio quede corriendo.

Para verificar que está funcionando:
```bash
psql --version
```

### Requisitos Python
```bash
pip install psycopg2-binary "duckdb>=0.10" pandas
```

### Generar los datos (puede tomar ~1 minuto)
```bash
python generate_data.py
```

Esto genera tres archivos CSV:
- `products.csv` (1,000 filas, 7 columnas)
- `customers.csv` (100,000 filas, 6 columnas)
- `sales.csv` (1,000,000 filas, 20 columnas — tabla ancha)

---

## 1. Cargar datos

```python
import os
import time
import psycopg2
import duckdb

# Ajustar esta ruta si es necesario (en Jupyter, __file__ no existe)
DATA_DIR = os.getcwd()  # Asume que el notebook está en la misma carpeta que los CSV
```

### PostgreSQL

```python
conn_pg = psycopg2.connect(dbname="clase03")
conn_pg.autocommit = True
cur = conn_pg.cursor()

# Crear tablas
cur.execute("""
DROP TABLE IF EXISTS sales;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
""")

cur.execute("""
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT,
    category     TEXT,
    brand        TEXT,
    base_price   NUMERIC(10,2),
    weight_kg    NUMERIC(10,2),
    rating       NUMERIC(3,1)
);

CREATE TABLE customers (
    customer_id       INTEGER PRIMARY KEY,
    customer_name     TEXT,
    region            TEXT,
    city              TEXT,
    segment           TEXT,
    registration_date DATE
);

CREATE TABLE sales (
    sale_id        INTEGER PRIMARY KEY,
    sale_date      DATE,
    sale_time      TIME,
    product_id     INTEGER,
    customer_id    INTEGER,
    quantity       INTEGER,
    unit_price     NUMERIC(10,2),
    discount_pct   NUMERIC(5,2),
    tax_pct        NUMERIC(5,2),
    amount         NUMERIC(12,2),
    payment_method TEXT,
    channel        TEXT,
    status         TEXT,
    shipping_cost  NUMERIC(10,2),
    shipping_days  INTEGER,
    warehouse_id   INTEGER,
    salesperson_id INTEGER,
    promotion_id   INTEGER,
    notes          TEXT,
    is_gift        BOOLEAN
);
""")

# Cargar CSVs (COPY es mucho más rápido que INSERT)
for table in ["products", "customers", "sales"]:
    csv_path = os.path.join(DATA_DIR, f"{table}.csv")
    with open(csv_path, "r") as f:
        cur.copy_expert(f"COPY {table} FROM STDIN WITH CSV HEADER NULL ''", f)

# Actualizar estadísticas del optimizador
cur.execute("ANALYZE;")
print("PostgreSQL: datos cargados")
```

### DuckDB

```python
conn_duck = duckdb.connect("clase03.duckdb")

for table in ["products", "customers", "sales"]:
    csv_path = os.path.join(DATA_DIR, f"{table}.csv")
    conn_duck.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_csv('{csv_path}')")

# DuckDB calcula estadísticas automáticamente — no necesita ANALYZE manual
print("DuckDB: datos cargados")
```

---

## 2. Sección PostgreSQL: Algoritmos de Join y Scan

### Funciones helper

```python
def explain_pg(query, analyze=True):
    """Ejecuta EXPLAIN (ANALYZE, BUFFERS) en PostgreSQL y muestra el resultado."""
    prefix = "EXPLAIN (ANALYZE, BUFFERS)" if analyze else "EXPLAIN"
    cur.execute(f"{prefix} {query}")
    for row in cur.fetchall():
        print(row[0])
    print()

def explain_duck(query, analyze=True):
    """Ejecuta EXPLAIN (ANALYZE) en DuckDB y muestra el resultado."""
    prefix = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
    result = conn_duck.execute(f"{prefix} {query}").fetchall()
    for row in result:
        print(row[-1] if len(row) > 1 else row[0])
    print()
```

### Nota sobre BUFFERS en la salida de EXPLAIN

La salida incluye líneas como `Buffers: shared hit=X read=Y`:
- **shared hit**: páginas que ya estaban en la memoria (buffer cache) de PostgreSQL
- **read**: páginas que tuvieron que leerse del disco

La segunda vez que ejecutan la misma consulta puede ser más rápida porque los datos quedan en cache. Para comparaciones justas, miren la primera ejecución.

---

### 2.1 Sequential Scan vs Index Scan

Primero, sin índices (aparte de las primary keys):

```python
# Consulta poco selectiva — ¿qué scan elige?
explain_pg("SELECT * FROM sales WHERE amount > 100;")
```

```python
# Consulta MUY selectiva — ¿qué scan elige?
explain_pg("SELECT * FROM sales WHERE sale_id = 42;")
```

Ahora, creemos un índice y observemos el cambio:

```python
cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);")

# Consulta selectiva: solo 1 día de 3 años (~0.1% de filas)
explain_pg("""
    SELECT * FROM sales
    WHERE sale_date = '2025-06-15';
""")
```

**Pregunta**: ¿Por qué el optimizador elige Seq Scan para `amount > 100` pero Index Scan para una fecha específica? El umbral de selectividad depende del ancho de la fila — con 20 columnas, PostgreSQL prefiere un scan secuencial incluso para selectividades moderadas.

### 2.2 Hash Join

```python
explain_pg("""
    SELECT p.category, SUM(s.amount)
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category
    ORDER BY SUM(s.amount) DESC;
""")
```

**Observar**: ¿Sobre qué tabla construye la hash table? ¿Por qué esa y no la otra?

### 2.3 Nested Loop Join (forzado)

```python
cur.execute("SET enable_hashjoin = off;")
cur.execute("SET enable_mergejoin = off;")

explain_pg("""
    SELECT p.category, SUM(s.amount)
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category
    ORDER BY SUM(s.amount) DESC;
""")

# Restaurar
cur.execute("RESET enable_hashjoin;")
cur.execute("RESET enable_mergejoin;")
```

**Nota**: Estamos forzando al optimizador — **nunca hagan esto en producción**. Lo hacemos para ver cómo funciona cada algoritmo.

**Pregunta**: ¿Cómo cambió el tiempo de ejecución respecto al Hash Join? ¿Por qué?

### 2.4 Sort-Merge Join (forzado)

```python
# Desactivar hash join Y nested loop para forzar merge join
cur.execute("SET enable_hashjoin = off;")
cur.execute("SET enable_nestloop = off;")

explain_pg("""
    SELECT c.region, SUM(s.amount), COUNT(*)
    FROM sales s
    JOIN customers c ON s.customer_id = c.customer_id
    GROUP BY c.region
    ORDER BY SUM(s.amount) DESC;
""")

cur.execute("RESET enable_hashjoin;")
cur.execute("RESET enable_nestloop;")
```

**Observar**: ¿Usa external merge sort o quicksort? ¿Para cuál de las dos tablas? ¿Cuánta memoria/disco usa el sort?

### 2.5 Aggregation

```python
explain_pg("""
    SELECT product_id,
           EXTRACT(MONTH FROM sale_date) AS month,
           SUM(amount), COUNT(*), AVG(quantity)
    FROM sales
    GROUP BY product_id, month
    ORDER BY SUM(amount) DESC
    LIMIT 20;
""")
```

**Observar**: ¿Usa HashAggregate o GroupAggregate? ¿Por qué?

---

## 3. Sección DuckDB: Column Store en acción

### 3.1 Las mismas consultas, diferente motor

```python
# Join + aggregation
explain_duck("""
    SELECT p.category, SUM(s.amount)
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category
    ORDER BY SUM(s.amount) DESC;
""")
```

```python
# Aggregation con GROUP BY de alta cardinalidad
explain_duck("""
    SELECT product_id,
           EXTRACT(MONTH FROM sale_date) AS month,
           SUM(amount), COUNT(*), AVG(quantity)
    FROM sales
    GROUP BY product_id, month
    ORDER BY SUM(amount) DESC
    LIMIT 20;
""")
```

**Observar**: ¿Qué operadores usa DuckDB? ¿En qué se diferencia del plan de PostgreSQL?

### 3.2 El test columnar: projection narrow vs wide

```python
# --- DuckDB ---
# Warm-up (para que la comparación sea justa)
conn_duck.execute("SELECT AVG(amount) FROM sales WHERE sale_date >= '2025-01-01';").fetchall()
conn_duck.execute("SELECT * FROM sales WHERE sale_date >= '2025-01-01';").fetchall()

# Solo 1 columna — el column store lee solo esa columna
t0 = time.time()
conn_duck.execute("SELECT AVG(amount) FROM sales WHERE sale_date >= '2025-01-01';").fetchall()
t_narrow_duck = time.time() - t0

# Todas las columnas — tiene que leer todo
t0 = time.time()
conn_duck.execute("SELECT * FROM sales WHERE sale_date >= '2025-01-01';").fetchall()
t_wide_duck = time.time() - t0

print(f"DuckDB  - 1 columna:    {t_narrow_duck:.3f}s")
print(f"DuckDB  - 20 columnas:  {t_wide_duck:.3f}s")
print(f"Ratio: {t_wide_duck/t_narrow_duck:.1f}x más lento con SELECT *")
```

Ahora lo mismo en PostgreSQL:

```python
# --- PostgreSQL ---
# Warm-up
cur.execute("SELECT AVG(amount) FROM sales WHERE sale_date >= '2025-01-01';")
cur.fetchall()
cur.execute("SELECT * FROM sales WHERE sale_date >= '2025-01-01';")
cur.fetchall()

t0 = time.time()
cur.execute("SELECT AVG(amount) FROM sales WHERE sale_date >= '2025-01-01';")
cur.fetchall()
t_narrow_pg = time.time() - t0

t0 = time.time()
cur.execute("SELECT * FROM sales WHERE sale_date >= '2025-01-01';")
cur.fetchall()
t_wide_pg = time.time() - t0

print(f"Postgres - 1 columna:    {t_narrow_pg:.3f}s")
print(f"Postgres - 20 columnas:  {t_wide_pg:.3f}s")
print(f"Ratio: {t_wide_pg/t_narrow_pg:.1f}x más lento con SELECT *")
```

**Pregunta**: ¿Por qué el ratio es mucho mayor en DuckDB que en PostgreSQL?

### 3.3 Point lookup: donde gana el row store

```python
# Warm-up en ambos motores
cur.execute("SELECT * FROM sales WHERE sale_id = 123456;")
cur.fetchall()
conn_duck.execute("SELECT * FROM sales WHERE sale_id = 123456;").fetchall()

# Timed: PostgreSQL
t0 = time.time()
cur.execute("SELECT * FROM sales WHERE sale_id = 123456;")
cur.fetchall()
t_pg_point = time.time() - t0

# Timed: DuckDB
t0 = time.time()
conn_duck.execute("SELECT * FROM sales WHERE sale_id = 123456;").fetchall()
t_duck_point = time.time() - t0

print(f"Point lookup - PostgreSQL: {t_pg_point*1000:.1f}ms")
print(f"Point lookup - DuckDB:     {t_duck_point*1000:.1f}ms")
```

**Pregunta**: ¿Por qué PostgreSQL gana en este caso? ¿Qué tipo de índice está usando?


### 4 Parquet: inspección de metadata

```python
# Exportar a Parquet
conn_duck.execute("COPY sales TO 'sales.parquet' (FORMAT PARQUET);")

result = conn_duck.execute("""
    SELECT 
        row_group_id,
        row_group_num_rows,
        path_in_schema AS column_name,
        total_compressed_size,
        total_uncompressed_size
    FROM parquet_metadata('sales.parquet')
    LIMIT 40;
""").fetchdf()

print(result.to_string())
```

**Observar**: ¿Cuántos row groups se generaron? ¿Qué columnas comprimen mejor (mayor ratio compressed/uncompressed)? ¿Por qué?

```python
# Consulta directa sobre Parquet (sin "cargar" la tabla)
explain_duck("SELECT AVG(amount) FROM 'sales.parquet' WHERE sale_date >= '2025-06-01';")
```

**Observar**: ¿El plan muestra que se saltan row groups gracias a las estadísticas min/max?

---

## 5. Preguntas para pensar

Responder en base a lo observado durante el ejercicio:

1. **¿Por qué PostgreSQL y DuckDB eligen algoritmos de join diferentes para la misma consulta?**
   Considerar: layout de almacenamiento, modelo de ejecución (tuple-at-a-time vs vectorizado), y diferencias en los optimizadores.

2. **¿Qué pasa con el rendimiento de DuckDB cuando hacen SELECT * vs SELECT de 2 columnas? ¿Y PostgreSQL?**
   Explicar la diferencia en los ratios observados.

3. **Ya crearon un índice en `sale_date` en la Sección 2.1. Revisando ese resultado: ¿podrían hacer lo mismo en DuckDB? ¿Por qué sí/no?**
   Hint: ¿Qué mecanismo usa DuckDB en lugar de índices B-tree?

4. **Miren la tabla de resumen de tiempos (Sección 3.4). ¿Cuándo gana cada motor y por qué?**
   Relacionar con la discusión OLTP vs OLAP de las slides.

5. **¿Dónde ayuda la compresión columnar? ¿Cuándo seguirían eligiendo un row store?**
   Den un ejemplo concreto de cada caso.

6. **Índices compuestos**

```python
# ¿Qué pasa si crean un índice compuesto?
cur.execute("CREATE INDEX IF NOT EXISTS idx_sales_prod_date ON sales(product_id, sale_date);")
explain_pg("""
    SELECT * FROM sales
    WHERE product_id = 42 AND sale_date >= '2025-01-01';
""")
```

7. **Órden de los datos**

```python
# ¿DuckDB puede aprovechar el orden de los datos?
conn_duck.execute("""
    COPY (SELECT * FROM sales ORDER BY sale_date)
    TO 'sales_sorted.parquet' (FORMAT PARQUET);
""")
result = conn_duck.execute("""
    SELECT row_group_id, column_name,
           stats_min, stats_max
    FROM parquet_metadata('sales_sorted.parquet')
    WHERE column_name = 'sale_date';
""").fetchdf()
print(result.to_string())
# ¿Cambian las estadísticas min/max? ¿Se pueden saltar más row groups ahora?
```

---

## Cleanup

```python
cur.close()
conn_pg.close()
conn_duck.close()
```
