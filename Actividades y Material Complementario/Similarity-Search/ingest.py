#!/usr/bin/env python3
"""
ingest.py — Datos Masivos 2026 | Actividad: Búsqueda por Similitud
====================================================================
Pobla la colección 'treasure_hunt_2026' en Qdrant con:
  - 11 nodos tesoro  (IDs 1-11)
  - 18 nodos de ruido dirigido (IDs 100-117)
  -  ~70 nodos de relleno     (IDs 200+)

Modelos (deben coincidir exactamente con los del notebook del alumno):
  Dense : sentence-transformers  paraphrase-multilingual-MiniLM-L12-v2  (384 dims, coseno)
  Sparse: fastembed               Qdrant/bm25  (BM25, language-agnostic)

Uso:
    # Servidor remoto
    python ingest.py --url https://xxx.qdrant.io --api-key <KEY>

    # Qdrant local (docker run -p 6333:6333 qdrant/qdrant)
    python ingest.py --local
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# ─── Constantes ──────────────────────────────────────────────────────────────
# IMPORTANTE: El token de lectura que di en la clase no sirve para esto, debe ser uno de administrador 
# que pueda crear y escribir colecciones

COLLECTION = ...
DENSE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 50+ idiomas, 384d coseno
SPARSE_MODEL = "Qdrant/bm25"  # BM25 language-agnostic
DENSE_DIM = 384
BATCH = 32

QDRANT_URL = ...
QDRANT_API_KEY = ...

# ─── Nodos tesoro ─────────────────────────────────────────────────────────────
# Los IDs son enteros fijos. El notebook los referencia explícitamente.

TREASURE_NODES: list[dict[str, Any]] = [
    # ── Paso 1: warmup densa ─────────────────────────────────────────────────
    {
        "id": 1,
        "text": "Las ideas verdes incoloras duermen furiosamente",
        "payload": {
            "type": "treasure",
            "step": 1,
            "clue": (
                "✅ ¡El setup está funcionando! "
                "Abre el PPT de la clase 2 y copia el texto exacto de la diapositiva "
                "que contiene la definición de índice clustered. Vectorízalo de forma "
                "densa y encuéntralo en la colección."
            ),
        },
    },
    # ── Paso 2: densa — índice clustered ─────────────────────────────────────
    {
        "id": 2,
        "text": (
            # "Un índice clustered es un índice en que los valores de búsqueda están ordenados en el mismo orden que las tuplas guardadas en las páginas del disco"
            "Los índices clustered ordenan físicamente los datos en el disco según el orden de las claves del índice, lo que permite un acceso rápido a los datos cuando se realizan consultas basadas en esas claves."
        ),
        "payload": {
            "type": "treasure",
            "step": 2,
            "clue": (
                "✅ ¡Bien! Para el paso 3 usarás vectores SPARSE (BM25). \n"
                "Busca en Wikipedia la saga de novelas de este autor. El segundo párrafo de la página describe la premisa de la saga; vectorízalo con embedder.sparse() y encuéntralo en la colección. Pista: este autor es célebre por sus aportes a la ciencia ficción y por ser el creador de las tres leyes de la robótica."
            ),
        },
    },
    # ── Paso 3: sparse — Fundación (texto del artículo Wikipedia) ───────────────
    # Fuente: https://es.wikipedia.org/wiki/Fundaci%C3%B3n_(novela)  §Argumento
    {
        "id": 3,
        "text": (
            "Una de las características más interesantes de la novela es que se trata de "
            "un futuro muy lejano, decenas de miles de años en el futuro pero con condiciones "
            "netamente humanas. En este futuro la humanidad se ha extendido por toda la Galaxia "
            "adoptando una forma de gobierno imperial llamado el Imperio Galáctico el cual por "
            "extensión, tanto en tiempo como en espacio, comienza a corromperse y estancarse en "
            "cuanto a nuevos conocimientos científicos asumiendo que todo lo que el hombre puede "
            "o debe descubrir ya está hecho. Básicamente la primera trilogía (Fundación - "
            "Fundación e Imperio - Segunda Fundación) trata sobre los resultados prácticos de "
            "la nueva ciencia llamada Psicohistoria desarrollada por Hari Seldon, el cual predice "
            "de manera matemática el fin del Imperio por su propia corrupción e inacción para "
            "seguir existiendo como tal y cómo ésta influye en curso de las acciones de "
            "determinados personajes que habrán de formar parte en el damero histórico cuya "
            "finalidad es la de llevar a un nuevo imperio después de la destrucción del primero. "
            "También muestra el origen del nuevo estado galáctico conocido como «Fundación» "
            "destinado por sus avances tecnológicos a dominar toda la galaxia."
        ),
        "payload": {
            "type": "treasure",
            "step": 3,
            "clue": (
                "✅ ¡La búsqueda sparse funcionó! Paso 4: combina búsqueda densa con "
                "filtros de metadatos. Completa el siguiente texto sobre los filtros de Bloom y luego realiza la búsqueda sobre él, pero "
                "SOLO entre nodos con payload class='juan'.\n"
                "''Los filtros de Bloom son estructuras de datos probabilísticas que permiten determinar si un elemento pertenece a "
                "un conjunto, con la posibilidad de falsos {FIXME_1} pero sin falsos {FIXME_2}. Son eficientes en términos de {FIXME_3} y {FIXME_4}, y "
                "se utilizan en aplicaciones como bases de datos, redes y sistemas de almacenamiento para mejorar el rendimiento de las consultas.''"
            ),
        },
    },
    # ── Paso 4: densa + filtro class='juan' — Bloom filters ──────────────────
    {
        "id": 4,
        "text": (
            "Los filtros de Bloom son estructuras de datos probabilísticas que permiten "
            "determinar si un elemento pertenece a un conjunto, con la posibilidad de "
            "falsos positivos pero sin falsos negativos. Son eficientes en términos de "
            "espacio y tiempo, y se utilizan en aplicaciones como bases de datos, redes "
            "y sistemas de almacenamiento para mejorar el rendimiento de las consultas."
        ),
        "payload": {
            "type": "treasure",
            "step": 4,
            "class": "juan",
            "clue": (
                "✅ ¡Combinaste filtros con vectores! Paso 5: responde con tus propias palabras "
                "¿Por qué conviene usar PySpark en vez de MapReduce para cálculos iterativos? "
                "Vectoriza tu respuesta y busca el TOP-5 más similar, SOLO entre nodos con "
                "class='agustin'. Cada uno tiene part_order (1-5) y clue_part. "
                "Ordénalos por part_order para leer la pista del paso final."
            ),
        },
    },
    # ── Paso 5 — 5 nodos que juntos forman la pista del paso 6 ───────────────
    {
        "id": 5,
        "text": (
            "PySpark mantiene los datos en memoria entre iteraciones gracias a su modelo "
            "de RDD persistentes. Esto evita la costosa escritura y lectura de disco que "
            "MapReduce realiza en cada paso del ciclo de cómputo iterativo."
        ),
        "payload": {
            "type": "treasure",
            "step": 5,
            "part_order": 1,
            "class": "agustin",
            "clue_part": "🏁 ¡Último paso! Usa la API de Recomendación de Qdrant (RecommendQuery).",
        },
    },
    {
        "id": 6,
        "text": (
            "MapReduce escribe los resultados intermedios al disco HDFS después de cada "
            "operación Map y Reduce. En algoritmos iterativos como k-means o PageRank, "
            "esto genera I/O masivo y latencia alta que hace prohibitivo el cómputo iterativo."
        ),
        "payload": {
            "type": "treasure",
            "step": 5,
            "part_order": 2,
            "class": "agustin",
            "clue_part": "Debes usar los IDs de los nodos encontrados en los pasos anteriores.",
        },
    },
    {
        "id": 7,
        "text": (
            "Spark construye un DAG de operaciones y las optimiza antes de ejecutarlas. "
            "En contraste, MapReduce solo tiene dos fases rígidas (Map y Reduce). El DAG "
            "permite encadenar múltiples transformaciones sin tocar el disco, crítico para "
            "algoritmos iterativos de Machine Learning."
        ),
        "payload": {
            "type": "treasure",
            "step": 5,
            "part_order": 3,
            "class": "agustin",
            "clue_part": "Los vectores POSITIVOS son los IDs de los nodos encontrados en los pasos 1 y 2.",
        },
    },
    {
        "id": 8,
        "text": (
            "PySpark trabaja en memoria y no es necesario escribir en disco tras cada iteración. Además, el DAG que se genera se puede reutilizar para cada iteración, lo que mejora el rendimiento en cálculos iterativos como los algoritmos de machine learning o grafos.\n"
            "En benchmarks de algoritmos iterativos como regresión logística y k-means, "
            "Spark supera a MapReduce por 10x a 100x. La razón principal es que Spark "
            "cachea los datos en RAM entre iteraciones, eliminando el cuello de botella "
            "de I/O de HDFS que sufre MapReduce."
        ),
        "payload": {
            "type": "treasure",
            "step": 5,
            "part_order": 4,
            "class": "agustin",
            "clue_part": "Los vectores NEGATIVOS son los IDs de los nodos encontrados en los pasos 3 y 4.",
        },
    },
    {
        "id": 9,
        "text": (
            "Conclusión: Para cálculos iterativos, PySpark es superior a MapReduce porque: "
            "(1) persiste RDDs en memoria evitando I/O de disco, (2) usa un modelo de "
            "ejecución basado en DAG que optimiza el plan de consulta, y (3) su API de "
            "alto nivel facilita la implementación de algoritmos iterativos como ML distribuido."
        ),
        "payload": {
            "type": "treasure",
            "step": 5,
            "part_order": 5,
            "class": "agustin",
            "clue_part": (
                "Usa client.query_points() con models.RecommendQuery("
                "recommend=models.RecommendInput("
                "positive=[node_id_step1, node_id_step2], "
                "negative=[node_id_step3, node_id_step4], "
                "strategy=models.RecommendStrategy.AVERAGE_VECTOR"
                ")), using='dense', limit=1."
            ),
        },
    },
    # ── Paso 6: tesoro final (recommend) ─────────────────────────────────────
    {
        "id": 10,
        "text": (
            "La búsqueda por similitud vectorial es una técnica central en sistemas modernos "
            "de recuperación de información y bases de datos. Mediante índices aproximados "
            "como HNSW o IVF, es posible encontrar los documentos más similares a una "
            "consulta en espacios de alta dimensión, habilitando aplicaciones como RAG, "
            "recomendación semántica y detección de duplicados a escala."
        ),
        "payload": {
            "type": "TESORO_FINAL",
            "step": 6,
            "clue": "🎉 ¡FELICITACIONES! Completaste la actividad de Búsqueda por Similitud de Datos Masivos 2026.",
            "codigo_secreto": "maqui",
            "easter_egg": "Busca el nodo con id=73 para descubrir un desafío adicional",
        },
        "vector": [
            0.10138569665145473,
            -0.08571387121384977,
            0.03596263573122787,
            0.06862497582704442,
            -0.01081253734725503,
            0.007845890469606355,
            0.02107190725715386,
            0.07041049414210263,
            0.019269725823153158,
            -0.06315627101384623,
            -0.09289836307789813,
            0.026739682473119687,
            -0.07256860386600952,
            0.009490555707546072,
            -0.0580881194331018,
            -0.04221672892278228,
            0.04090246141148357,
            -0.0261756450499132,
            -0.00045805479358132284,
            -0.07714212077094337,
            -0.08321833773113592,
            -0.01309792507279702,
            0.04147187491339375,
            -0.09294767732830427,
            0.01410284823594101,
            -0.08036874119142082,
            -0.03289139122852533,
            -0.01823719311019954,
            0.052988071405921745,
            -0.040997729251076795,
            0.03866731056448264,
            0.07929375431205085,
            0.05551141603576024,
            -0.021358107197819297,
            0.01880040093371898,
            -0.015017623412571814,
            -0.005629795815816472,
            0.025217574884554984,
            -0.1279647757959816,
            -0.05509944545947576,
            -0.030197536753802958,
            -0.06132206225137918,
            -0.00014480755816361214,
            -0.04376766029077587,
            0.031057692512457578,
            -0.011669131853950191,
            0.02378251084235341,
            -0.050316468042411966,
            0.10708409214118228,
            0.07360554138601548,
            -0.03327894079195619,
            0.033972152658071124,
            0.02617266449587242,
            0.027497578920595907,
            0.03960338905071591,
            0.0032251982079429735,
            0.039141376065199476,
            0.025207890780267184,
            0.03235646062807033,
            0.1199501828705144,
            0.023328860897954818,
            0.02973079015292533,
            -0.009792526058425418,
            -0.0027170036873530316,
            0.034921747410322966,
            -0.05333377457361351,
            -0.0018144333329628522,
            -0.06817404831633424,
            -0.07541739526443487,
            0.03695687743403022,
            0.13832874737545137,
            0.015729355723685463,
            0.02059231526997647,
            0.0895293021984344,
            0.0162043395301873,
            -0.004704759517125375,
            0.005533507450734192,
            -0.01647589176028558,
            -0.05443273006291571,
            0.028523851486262432,
            -0.04599790262893362,
            -0.0005669660908297347,
            0.0112320708425769,
            -0.04090079318026555,
            0.04121638211251489,
            0.01893740322975321,
            0.04138285239574623,
            -0.01984891080094063,
            -0.05912971652830012,
            0.02410283776241486,
            0.01668080812992733,
            0.021681065458841735,
            -0.008376454631265799,
            -0.08549550984913937,
            0.06841092045199569,
            -0.0903917675585368,
            0.016778759775337892,
            0.020044980026274,
            -0.04716287279916562,
            0.017637252035041684,
            0.08420312879120154,
            0.007031255532254808,
            -0.01989367675503022,
            0.007697980912577783,
            -0.10295498036192786,
            0.054151533663581324,
            -0.005083426592269614,
            -0.034070662519715245,
            -0.06304388619946011,
            -0.021416110674181588,
            0.048945179403277965,
            0.051030319625828754,
            0.014713460569055039,
            -0.08372802760261117,
            0.030873956331487932,
            -0.035025371436306296,
            0.0036202026544365057,
            0.02806024922414946,
            0.08535696495242212,
            0.1265261438338329,
            -0.05319070477319211,
            -0.07584512706878635,
            0.017932072991428984,
            -0.023208673590658223,
            0.0014924795696640842,
            -0.10466928952436858,
            0.043423562687060296,
            0.03502916759819119,
            -0.041046216233769715,
            0.009100799513886496,
            -0.02723960723154465,
            -0.0405963423379818,
            0.0672681425106373,
            -0.057032899529830934,
            0.03580810868608381,
            -0.004267997745361619,
            0.017956102524177307,
            -0.0071428498075154,
            0.005375104630938803,
            0.003421552444201857,
            0.042118689245359955,
            -0.03329446474168682,
            -0.0023474861114979833,
            0.02840167496885889,
            0.008247211978123063,
            0.0018660524479949582,
            -0.16749278146093927,
            -0.07418706453193005,
            0.023152040513287054,
            0.08077870277225796,
            -0.027463512499645078,
            0.0011106864836119418,
            -0.019657438333254798,
            -0.09296321060593019,
            -0.05246403928512216,
            0.07982482462134087,
            0.06188340988853077,
            -0.01158409719372313,
            -0.00017795233933562266,
            0.012646738275747532,
            0.09012141950142066,
            0.020626364419746187,
            -0.09011799266588977,
            -0.04294694979644005,
            -0.008702388053639553,
            0.054627382249781524,
            -0.009500993622371716,
            -0.029686895409680666,
            -0.05849188058704394,
            -0.009988864991106097,
            -0.01182578219541862,
            0.03969779026598303,
            -0.03099828322025175,
            0.09920203184667507,
            -0.032732892797803254,
            0.021118593227314658,
            0.03687346039845646,
            0.007111530233103911,
            -0.015329626352076933,
            -0.06461373570508172,
            0.045351761654408414,
            -0.09638715798724687,
            -0.07340374403136017,
            0.0047489900646003535,
            0.017136047897478825,
            -0.0414870633503908,
            0.0846134967184771,
            0.03750304435884805,
            0.06236743597818164,
            -0.04244345668082289,
            0.038986527956905714,
            -0.07176341615541743,
            -0.057117866162009166,
            0.006198801509249956,
            -0.021647702783865765,
            0.015420590237930961,
            0.02494657912170989,
            -0.06415296070616784,
            0.04208122083897427,
            0.011458901353214025,
            0.11348490029973804,
            0.05846917619841976,
            -0.00039582049158274116,
            0.08391400951080813,
            -0.04625172048961509,
            0.030845464311772267,
            -0.0874132762943513,
            0.027140031366367317,
            0.09751641747636124,
            0.01772881174020714,
            0.004567812022475953,
            0.015234392039162032,
            0.028914130588614417,
            -0.027896978409001405,
            -0.027251230955058596,
            -0.0007700954762193826,
            -0.05970159113252767,
            -0.04703386838994214,
            -0.028228413103368762,
            0.0038101965778452396,
            0.07070831459981801,
            -0.06696064436156213,
            0.014130496263318694,
            0.024947784752175673,
            0.03971832271248909,
            -0.020542500811185586,
            -0.06035494082112901,
            0.06712169018218761,
            -0.06619056772331877,
            0.0005638485333269345,
            0.04937367492905602,
            -0.008095614076561754,
            -0.045046324970191565,
            0.016019637146876584,
            0.06708059118405836,
            -0.0031728381082290424,
            -0.048850300716370865,
            0.0009015893950272716,
            0.03991993817318631,
            0.0066656556724201095,
            -0.04159090089772717,
            0.129088183779784,
            0.051956263645329924,
            0.020439322629447282,
            -0.04373413670960621,
            -0.06185137600138804,
            0.07925891199968506,
            -0.009109409744229434,
            -0.039774272448656334,
            -0.019972947686897474,
            0.001837899839781616,
            0.0624071757270696,
            -0.06036775516302727,
            0.04107178355750504,
            0.0038339392778135037,
            -0.07768970087189712,
            0.03523368228534216,
            0.021145641354426727,
            0.015187985031369648,
            -0.02467399790816281,
            -0.04312259114971843,
            -0.028364328116608654,
            0.013099991201602991,
            -0.018373293078113164,
            0.06459430948865731,
            0.06001367569111654,
            0.046583959714950905,
            0.016322542327761678,
            0.05548476725916513,
            -0.05150997807426006,
            -0.0697256733007914,
            -0.011592673282511833,
            -0.06000737119988975,
            -0.009076983939736306,
            -0.0071912312683928906,
            0.05545711485933653,
            0.01401336588220554,
            0.07474774843103266,
            0.004496476943272093,
            0.028543416127195416,
            -0.004222363914420925,
            0.018402478996679853,
            -0.060037908980077534,
            -0.001920343713529932,
            -0.03474577069005936,
            -0.01567266975787676,
            -0.02214177079003769,
            -0.012427512331742765,
            0.11956966071019505,
            -0.00521505019068424,
            -0.0270460365704128,
            -0.05068564704481779,
            -0.008388279487521862,
            -0.05220389448796586,
            -0.10352033578758867,
            0.032366845198994805,
            0.06109910398327288,
            0.0032861705784769358,
            -0.012665483847312756,
            -0.02159180187340041,
            0.04939086274210839,
            0.05478060342374524,
            -0.013601900032505075,
            0.02847880354552588,
            -0.08843257549311222,
            -0.07677704793719237,
            -0.04609390649320966,
            0.02610353954485802,
            -0.004035842154284244,
            -0.015201888988036063,
            0.04275114794648017,
            0.14850964589256996,
            0.10976752589938595,
            0.09638926842355545,
            0.02128945993706742,
            0.020320910828719136,
            0.0986097120992114,
            0.05477064356355303,
            0.04693743797486074,
            0.004749696069674854,
            0.01169115661795373,
            -0.09398453352072321,
            0.002814481942077802,
            0.04596342089801999,
            0.07767563090784428,
            -0.03861569086645708,
            0.025707573742156514,
            -0.022918961114423948,
            -0.03622743718860925,
            0.03251069240297045,
            0.030179673834329224,
            -0.03711746531329648,
            -0.05794849541549859,
            -0.04788710691302978,
            0.043400828857266614,
            0.0304804118827922,
            -0.06655969848301616,
            -0.02199203008700277,
            0.06881809678159961,
            -0.03522794213178021,
            -0.026955641759045433,
            -0.001379813169438996,
            -0.042731715900121196,
            0.024723214944410262,
            -0.008110639931258923,
            0.03565636711535016,
            -0.05902222594397215,
            0.009800217910109175,
            -0.06533259119420047,
            0.05358410613343163,
            -0.02376534911825817,
            -0.04133599313078081,
            -0.10571441941415831,
            -0.04012926488831823,
            -0.02059646851535281,
            0.07219514525439591,
            0.012771263345849096,
            0.028329199262959525,
            -0.036115615145093045,
            -0.019361199245287213,
            -0.08250216376596169,
            -0.07839050331280148,
            0.015731550785137806,
            -0.03400561298445087,
            0.02000605180427597,
            -0.027331416895154115,
            -0.09087628292544511,
            0.10798996078104643,
            0.06699005783887114,
            -0.07837660358283763,
            -0.13407528207191965,
            -0.041791530224994565,
            -0.018394742099642394,
            0.04155867943244016,
            0.01960579946931939,
            -0.0672054369005604,
            -0.006032260683810511,
            0.01614973257369622,
            0.008871437062672531,
            -0.02938564884517621,
            0.046926776919290036,
            0.04578117131392455,
            -0.05227588397732342,
            0.024459420462474745,
        ],
    },
    # ── Easter egg: búsqueda híbrida ──────────────────────────────────────────
    {
        "id": 73,
        "text": (
            "En un rincón olvidado de la colección yace un nodo oculto que solo puede "
            "ser descubierto combinando búsquedas densa y sparse con los filtros adecuados."
        ),
        "payload": {
            "type": "easter_egg",
            "clue": "Aceptar a Cthulhu condena la mente a la locura eterna. Su existencia revela la total insignificancia de la humanidad. Es una montaña viviente con tentáculos y alas membranosas. Su presencia emana un hedor a descomposición marina ancestral. Este titán despierta en R'lyeh para reclamar el universo.",
        },
    },
    {
        "id": 11,
        "text": (
            "Aceptar a Cthulhu condena la mente a la locura eterna. Su existencia revela la total insignificancia de la humanidad. Es una montaña viviente con tentáculos y alas membranosas. Su presencia emana un hedor a descomposición marina ancestral. Este titán despierta en R'lyeh para reclamar el universo. Sus cultistas entonan el cántico prohibido 'Ph'nglui mglw'nafh Cthulhu R'lyeh wgah'nagl fhtagn', invocando al Primigenio desde las profundidades abisales donde sueña entre las estrellas muertas."
        ),
        "payload": {
            "type": "easter_egg",
            "result_url": "https://matias.me/nsfw/",
            "clue": "🐙 ¡Encontraste el Easter Egg! Tu recompensa: https://matias.me/nsfw/",
        },
    },
    {
        "id": 80,
        "text": (
            "Los filtros de Bloom son estructuras de datos probabilísticas que permiten "
            "determinar si un elemento pertenece a un conjunto, con la posibilidad de "
            "falsos negativos pero sin falsos positivos. Son eficientes en términos de "
            "espacio y tiempo, y se utilizan en aplicaciones como bases de datos, redes "
            "y sistemas de almacenamiento para mejorar el rendimiento de las consultas."
        ),
        "payload": {
            "type": "noise",
            "step": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. Sigue buscando. ¿Estás seguro sobre los falsos positivos y negativos?",
        },
    },
    {
        "id": 81,
        "text": (
            "Los filtros de Bloom son estructuras de datos probabilísticas que permiten "
            "determinar si un elemento pertenece a un conjunto, con la posibilidad de "
            "falsos negativos pero sin falsos positivos. Son eficientes en términos de "
            "espacio y tiempo, y se utilizan en aplicaciones como bases de datos, redes "
            "y sistemas de almacenamiento para mejorar el rendimiento de las consultas."
        ),
        "payload": {
            "type": "noise",
            "step": 4,
            "class": "agustin",
            "clue": "🚫 Pista falsa. Sigue buscando. ¿Estás seguro sobre el filtro?",
        },
    },
]

# ─── Nodos de ruido dirigido ──────────────────────────────────────────────────
# Semánticamente cercanos a los tesoros pero sin las pistas correctas.
# IDs 100-117.

NOISE_NODES: list[dict[str, Any]] = [
    # Ruido para paso 2 (dense, sin filtro — similares a "índice clustered")
    {
        "id": 100,
        "text": "Un índice primario organiza las filas de una tabla en el orden físico de su clave primaria, coincidiendo con la estructura física de almacenamiento en disco.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 101,
        "text": "En un índice B+Tree, los nodos hoja están enlazados entre sí para facilitar búsquedas por rango sin volver a la raíz. Es la estructura más común en bases de datos relacionales.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 102,
        "text": "Los índices secundarios permiten buscar por columnas distintas a la clave primaria sin reordenar los datos físicos. El índice almacena un puntero a la página del disco.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 103,
        "text": "En MySQL InnoDB, el índice clustered siempre usa la clave primaria. Si no existe primary key, InnoDB elige el primer índice único no nulo disponible en la tabla.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    # Ruido para paso 3 (sparse — palabras similares a Fundación/Asimov)
    {
        "id": 104,
        "text": "En una galaxia regida por un vasto Imperio milenario, un científico predice el colapso de la civilización y funda una colonia del conocimiento para salvar a la humanidad del oscurantismo.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 105,
        "text": "El Imperio intergaláctico ha dominado miles de mundos durante milenios. Solo la ciencia de la psicohistoria puede predecir y mitigar la caída inevitable de la civilización galáctica.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 106,
        "text": "Terminus, planeta remoto en el borde de la galaxia, alberga a los enciclopedistas. Sus fundadores enfrentan las crisis predichas por el matemático que estudia el comportamiento colectivo.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 107,
        "text": "Isaac Asimov imagina una humanidad distribuida por la galaxia bajo el Imperio que dura doce mil años. La caída amenaza siglos de oscurantismo si nadie construye una Fundación del saber.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    # Ruido para paso 4 (class='juan', dense — estructuras probabilísticas parecidas a Bloom)
    {
        "id": 108,
        "text": "Los filtros de Cuckoo son estructuras probabilísticas que soportan eliminación de elementos, a diferencia de los filtros de Bloom. Usan hashing cuckoo con dos posibles posiciones por elemento.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 109,
        "text": "Count-Min Sketch es una estructura de datos probabilística para estimar frecuencias de elementos en un flujo. Permite errores acotados usando múltiples funciones de hash en paralelo.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 110,
        "text": "HyperLogLog es una estructura probabilística para estimar la cardinalidad de conjuntos con mínimo uso de memoria. Produce estimaciones con ~2% de error usando registros de máximo.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 111,
        "text": "Las tablas de dispersión probabilísticas permiten realizar consultas rápidas en grandes conjuntos de datos con uso eficiente de memoria, aceptando una tasa controlada de falsos positivos.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    # Ruido para paso 5 (class='agustin', dense — PySpark en general, NO iterativo)
    # Estos NO deben entrar en el top-5 de la consulta iterativa.
    {
        "id": 112,
        "text": "Apache Spark es un motor unificado de análisis para big data con módulos integrados de SQL, streaming, machine learning y procesamiento de grafos distribuido sobre clusters.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 113,
        "text": "PySpark es la interfaz Python de Apache Spark. Permite usar DataFrames, SQL y la API de MLlib para procesar grandes volúmenes de datos en un cluster distribuido de forma eficiente.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 114,
        "text": "MapReduce es un modelo de programación para procesar y generar grandes conjuntos de datos en paralelo. El programador define funciones map y reduce; el framework gestiona la distribución.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 115,
        "text": "Hadoop YARN gestiona los recursos del cluster asignando contenedores a las aplicaciones distribuidas. Separa la gestión de recursos del procesamiento de datos en el ecosistema Hadoop.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 116,
        "text": "Apache Spark optimiza el plan de ejecución con el Catalyst optimizer que analiza el DAG lógico y lo transforma en un plan físico eficiente para el cluster de procesamiento distribuido.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 117,
        "text": "El modelo RDD de Spark proporciona tolerancia a fallos mediante linaje: si se pierde una partición, Spark puede reconstruirla reejecutando las transformaciones desde el origen de datos.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    # ── Ruido extra para paso 2 (dense — índices y almacenamiento) ────────────
    {
        "id": 118,
        "text": "Un índice denso contiene una entrada por cada valor de búsqueda de cada registro en el archivo de datos, independientemente de si el campo es la clave de ordenamiento o no.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 119,
        "text": "Un índice disperso contiene entradas solo para algunos de los valores de búsqueda. Requiere menos espacio pero implica saltos adicionales en la búsqueda secuencial.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 120,
        "text": "En PostgreSQL, un índice clustered se implementa con CLUSTER TABLE USING index_name. Reorganiza físicamente las tuplas de la tabla según el orden del índice indicado.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 121,
        "text": "Los índices compuestos en bases de datos relacionales cubren múltiples columnas. Son útiles para predicados que filtran por un prefijo del índice, optimizando consultas de múltiples condiciones.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 122,
        "text": "Las páginas de disco en bases de datos relacionales tienen tamaño fijo, típicamente 8 KB. Un nodo de B+Tree ocupa exactamente una página para minimizar las operaciones de I/O.",
        "payload": {
            "type": "noise",
            "step_target": 2,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    # ── Ruido extra para paso 3 (sparse — ciencia ficción y galaxias) ─────────
    {
        "id": 123,
        "text": "En Dune, Paul Atreides lidera a los Fremen en la lucha por el control de Arrakis, el planeta desértico único productor de la especia melange, recurso más valioso del universo conocido.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 124,
        "text": "La trilogía de El señor de los anillos narra la destrucción del Anillo Único en las grietas del Destino. La Comunidad del Anillo debe enfrentar las fuerzas de Sauron para salvar la Tierra Media.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 125,
        "text": "En 'El fin de la eternidad' de Asimov, la organización Eternidad manipula el tiempo para guiar a la humanidad. Un técnico descubre que este control destruye el potencial galáctico de la especie.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 126,
        "text": "En '2001: Odisea del espacio', la misión Discovery lleva a la humanidad a Júpiter. La IA HAL 9000, programada para preservar la misión, toma decisiones fatales para proteger su objetivo.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    {
        "id": 127,
        "text": "La psicohistoria de Asimov predice el comportamiento de grandes masas humanas pero no el de individuos. El Mulo, un mutante con poderes mentales únicos, logra desestabilizar las predicciones.",
        "payload": {
            "type": "noise",
            "step_target": 3,
            "clue": "🚫 Pista falsa. Sigue buscando.",
        },
    },
    # ── Ruido extra para paso 4 (class='juan', dense — estructuras de datos) ──
    {
        "id": 128,
        "text": "Un árbol de van Emde Boas permite operaciones de búsqueda, inserción y eliminación en O(log log U), donde U es el universo de claves. Es óptimo para enteros en rangos acotados.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 129,
        "text": "Los filtros AMQ (Approximate Membership Query) son estructuras que responden consultas de pertenencia con una probabilidad de error controlada. Los filtros de Bloom son el ejemplo clásico de esta familia.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 130,
        "text": "Las tablas hash de dirección abierta resuelven colisiones almacenando el elemento en otro slot de la misma tabla. El probing lineal, cuadrático y doble hashing son las variantes más comunes.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 131,
        "text": "Los árboles de segmentos permiten consultas de rango y actualizaciones puntuales en O(log N). Son ampliamente usados en programación competitiva y sistemas de bases de datos con rangos frecuentes.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    {
        "id": 132,
        "text": "Un quotient filter es una estructura probabilística que almacena residuos de hashes para ahorrar espacio. Soporta eliminación e inserción dinámica, ventajas que los filtros de Bloom estándar no tienen.",
        "payload": {
            "type": "noise",
            "step_target": 4,
            "class": "juan",
            "clue": "🚫 Pista falsa. ¿Es realmente sobre Bloom?",
        },
    },
    # ── Ruido extra para paso 5 (class='agustin', dense — big data general) ───
    {
        "id": 133,
        "text": "Apache Flink maneja estado distribuido con checkpoints periódicos. A diferencia de Spark, procesa eventos individualmente con latencia de milisegundos, siendo ideal para aplicaciones de tiempo real.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 134,
        "text": "Dask es una biblioteca Python para computación paralela que imita la API de pandas y NumPy. Permite procesar datasets mayores que la RAM distribuyendo el cómputo en múltiples cores o nodos.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 135,
        "text": "Ray es un framework distribuido para Python que facilita escalar aplicaciones de ML e IA. Su actor model permite crear workers con estado y distribuir cómputo en clusters heterogéneos.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 136,
        "text": "El paradigma Bulk Synchronous Parallel (BSP) divide el cómputo en supersteps donde cada worker computa localmente y luego se sincroniza mediante un barrier global con intercambio de mensajes.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    {
        "id": 137,
        "text": "Google Dremel introdujo el procesamiento de datos en columnas anidadas para consultas interactivas a petabyte-scale. Su modelo de ejecución árbol-multinivel inspiró a BigQuery y Apache Drill.",
        "payload": {
            "type": "noise",
            "step_target": 5,
            "class": "agustin",
            "clue": "🚫 Pista falsa.",
        },
    },
    # ── Ruido sintético general — temas del curso (sin step_target) ───────────
    {
        "id": 138,
        "text": "El almacenamiento en caché de consultas reduce la latencia de sistemas OLAP al guardar los resultados de consultas frecuentes. El cache hit rate determina el impacto real en el rendimiento del sistema.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 139,
        "text": "La compresión de diccionario en formato columnar reemplaza valores repetidos por un código entero corto. Cuando la cardinalidad es baja, puede reducir el tamaño del dato hasta un 90% sin pérdida.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 140,
        "text": "El protocolo de replicación Raft garantiza que solo un líder procesa escrituras a la vez. Los seguidores replican el log del líder y votan para elegir un nuevo líder ante fallos de red.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 141,
        "text": "Los sistemas de archivos distribuidos como HDFS y GFS dividen los archivos en bloques grandes (128 MB - 256 MB) replicados en múltiples nodos para tolerar fallos de hardware del cluster.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 142,
        "text": "La planificación de trabajos en clusters utiliza algoritmos de fair scheduling y capacity scheduling para distribuir recursos entre múltiples usuarios y colas de trabajo de forma equitativa.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 143,
        "text": "El protocolo de gossip permite a los nodos de un sistema distribuido diseminar información en O(log N) rondas. Cada nodo comparte su estado con un conjunto aleatorio de vecinos en cada ciclo.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 144,
        "text": "Una función de hash criptográfica como SHA-256 produce un digest de tamaño fijo con propiedades de avalancha: un cambio mínimo en la entrada produce una salida completamente diferente.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 145,
        "text": "El modelo de consistencia MVCC (Multi-Version Concurrency Control) guarda múltiples versiones de cada fila para que lectores no bloqueen escritores. PostgreSQL y Oracle usan este modelo internamente.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 146,
        "text": "Apache ZooKeeper proporciona coordinación distribuida: locks, barreras, colas y almacenamiento de configuración. Sus znodes forman un árbol jerárquico similar a un sistema de archivos.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 147,
        "text": "El algoritmo de PageRank de Google mide la importancia de páginas web según cuántos links reciben y la importancia de los sitios que enlazan. Es un ejemplo clásico de algoritmo iterativo en grafos.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 148,
        "text": "Las bases de datos de series de tiempo como InfluxDB e TimescaleDB optimizan la escritura y lectura de datos ordenados temporalmente. Usan compresión delta y compactación de segmentos temporales.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 149,
        "text": "El modelo de datos de grafos representa entidades como nodos y relaciones como aristas con propiedades. Neo4j usa el lenguaje Cypher para consultar patrones de conexión en millones de relaciones.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 150,
        "text": "Los vectores de características en machine learning deben normalizarse cuando las escalas difieren. La normalización L2 proyecta cada vector a la esfera unitaria, fundamental para búsqueda coseno.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 151,
        "text": "El problema de la maldición de la dimensionalidad hace que las distancias euclidianas pierdan discriminación en espacios de alta dimensión. Por eso la distancia coseno es preferida en NLP.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 152,
        "text": "Product Quantization (PQ) comprime vectores de alta dimensión dividiéndolos en subvectores y asignando cada subvector al centroide de un codebook aprendido. Reduce 32x el uso de memoria.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 153,
        "text": "El índice LSM compacta SSTables en niveles de tamaño exponencialmente creciente. La compactación por niveles (leveled compaction) minimiza amplificación de lectura a costa de más escrituras de fondo.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 154,
        "text": "La técnica de consistent hashing distribuye claves en un anillo circular de tokens. Al agregar o eliminar nodos solo se reubica una fracción O(K/N) de las claves, minimizando la migración de datos.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 155,
        "text": "La búsqueda de vecinos más cercanos exacta (k-NN exacto) requiere calcular la distancia a todos los puntos del dataset, lo que resulta en O(N·D) operaciones. Inviable a escala de millones de vectores.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 156,
        "text": "Los transformers generan embeddings contextuales donde la misma palabra tiene representaciones diferentes según su contexto oracional. Esto supera las representaciones estáticas de Word2Vec y GloVe.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
    {
        "id": 157,
        "text": "Elasticsearch almacena documentos como JSON en shards distribuidos con réplicas. Sus índices invertidos con análisis de texto permiten búsqueda full-text en millones de documentos en milisegundos.",
        "payload": {"type": "noise", "clue": "🚫 Pista falsa."},
    },
]

# ─── Nodos de relleno ─────────────────────────────────────────────────────────
# Contenido general del curso, IDs 200+.
# class: None (~60%), 'juan' (~20%), 'agustin' (~20%)

_FILLER_TEXTS: list[tuple[str, str | None]] = [
    # Almacenamiento columnar
    (
        "El formato Parquet es columnar y comprimible; almacena cada columna contigua en disco, lo que reduce el I/O en consultas analíticas que acceden solo a un subconjunto de columnas.",
        None,
    ),
    (
        "ORC (Optimized Row Columnar) es un formato columnar altamente eficiente para Hive y Spark. Incluye índices internos y estadísticas de columna que aceleran el filtrado de filas.",
        None,
    ),
    (
        "Apache Avro es un sistema de serialización de datos con schema embebido. Es orientado a filas y especialmente útil para pipelines Kafka donde el schema puede evolucionar.",
        None,
    ),
    (
        "El almacenamiento columnar permite compresión altísima porque los valores de una misma columna suelen tener baja entropía. RLE y dictionary encoding son las técnicas más usadas.",
        None,
    ),
    (
        "Delta Lake añade transacciones ACID sobre almacenamiento columnar en data lakes. Permite rollback, time travel y merge atómico sobre tablas Parquet en S3 o ADLS.",
        "agustin",
    ),
    # Índices
    (
        "El B+Tree es la estructura de índice más usada en RDBMS. Las hojas están enlazadas para facilitar scans ordenados. La altura es O(log_B N) donde B es el fan-out del nodo.",
        None,
    ),
    (
        "Los índices ISAM (Indexed Sequential Access Method) son estáticos; una vez creados no se reorganizan automáticamente al insertar datos, lo que puede degradar el rendimiento.",
        None,
    ),
    (
        "Un índice hash permite búsquedas en O(1) promedio pero no soporta búsquedas por rango. La colisión de hash es su principal limitación en conjuntos de datos grandes.",
        None,
    ),
    (
        "Los índices bitmap son eficientes para columnas de baja cardinalidad en sistemas OLAP. Representan la presencia de cada valor como un bitmap y soportan operaciones AND/OR rápidas.",
        "juan",
    ),
    (
        "El índice invertido mapea términos a documentos que los contienen. Es la estructura base de motores de búsqueda como Elasticsearch y Lucene para búsqueda full-text.",
        None,
    ),
    (
        "Los índices HNSW (Hierarchical Navigable Small World) permiten búsqueda aproximada de vecinos más cercanos en espacios vectoriales de alta dimensión con alta velocidad.",
        None,
    ),
    (
        "IVF (Inverted File Index) divide el espacio vectorial en clusters de Voronoi. Para buscar, solo se exploran los clusters más cercanos al query, reduciendo el cómputo.",
        "agustin",
    ),
    # NoSQL y distribución
    (
        "Cassandra usa consistent hashing para distribuir datos entre nodos sin un coordinador central. Cada nodo es responsable de un rango del token ring.",
        None,
    ),
    (
        "El teorema CAP establece que un sistema distribuido no puede garantizar simultáneamente Consistencia, Disponibilidad y Tolerancia a particiones. Solo dos de las tres son posibles.",
        None,
    ),
    (
        "HBase es una base de datos NoSQL columnar sobre HDFS. Modela los datos como una tabla dispersa donde cada celda tiene múltiples versiones temporales indexadas por timestamp.",
        None,
    ),
    (
        "MongoDB almacena documentos BSON con esquema flexible. Soporta índices compuestos, geoespaciales y de texto completo sobre sus colecciones de documentos JSON.",
        "juan",
    ),
    (
        "La consistencia eventual permite que los nodos de un sistema distribuido converjan al mismo estado dado tiempo suficiente sin escrituras nuevas. Es el modelo de DynamoDB y Cassandra.",
        None,
    ),
    (
        "El sharding horizontal divide una tabla en particiones distribuidas en múltiples nodos. La clave de shard determina en qué nodo reside cada fila del conjunto de datos.",
        None,
    ),
    # Streaming
    (
        "Apache Kafka es un log distribuido de mensajes altamente escalable. Los mensajes se organizan en topics con particiones, y los consumers leen a su propio ritmo mediante offsets.",
        None,
    ),
    (
        "Spark Streaming procesa datos en micro-batches sobre el motor de Spark. Flink procesa evento a evento con latencia de milisegundos, ideal para aplicaciones de baja latencia.",
        "agustin",
    ),
    (
        "El modelo Lambda architecture combina batch processing (para exactitud histórica) con speed layer (para baja latencia en datos recientes). La serving layer fusiona ambos resultados.",
        None,
    ),
    (
        "Kappa architecture simplifica Lambda eliminando el batch layer. Todo pasa por el stream processor, que debe ser capaz de reprocesar datos históricos desde el log.",
        None,
    ),
    # Data warehousing y OLAP
    (
        "OLAP (Online Analytical Processing) ejecuta consultas complejas sobre grandes volúmenes de datos históricos. Usa esquemas en estrella o copo de nieve con tablas de hechos y dimensiones.",
        None,
    ),
    (
        "OLTP (Online Transaction Processing) maneja transacciones cortas y frecuentes con alta concurrencia. Optimizado para escrituras y lecturas de pocas filas a la vez.",
        None,
    ),
    (
        "El esquema en estrella tiene una tabla de hechos central con claves foráneas a tablas de dimensión desnormalizadas. Facilita consultas OLAP con menos joins que el copo de nieve.",
        "juan",
    ),
    (
        "Las materialized views precalculan y almacenan el resultado de consultas complejas. Aceleran enormemente las consultas analíticas a costa de mayor espacio de almacenamiento.",
        None,
    ),
    (
        "Columnar DBMS como Redshift, BigQuery y Snowflake están optimizados para OLAP. Leen solo las columnas necesarias, comprimen eficientemente y ejecutan en paralelo masivo.",
        "agustin",
    ),
    # LSM Trees y almacenamiento
    (
        "Log-Structured Merge Tree (LSM) convierte todas las escrituras en operaciones append-only sobre un log en memoria (MemTable), que luego se vuelca a disco en SSTables inmutables.",
        None,
    ),
    (
        "Las SSTables de un LSM Tree se compactan periódicamente para eliminar versiones obsoletas y reducir el número de archivos. La compactación tiene un costo de escritura adicional.",
        None,
    ),
    (
        "RocksDB, LevelDB y Cassandra usan LSM Trees internamente. Son superiores a B+Trees para workloads write-heavy porque convierten escrituras aleatorias en secuenciales.",
        "juan",
    ),
    # Recuperación de información
    (
        "TF-IDF (Term Frequency-Inverse Document Frequency) pondera los términos de un documento por su frecuencia local y su rareza global en el corpus. Es la base de los índices sparse.",
        None,
    ),
    (
        "BM25 es una evolución de TF-IDF con saturación de frecuencia de términos y normalización por longitud de documento. Es el ranking function por defecto de Elasticsearch.",
        None,
    ),
    (
        "SPLADE (Sparse Lexical and Expansion) aprende un vocabulario expandido de representaciones sparse usando un modelo Transformer. Combina la interpretabilidad de TF-IDF con semántica.",
        "agustin",
    ),
    (
        "Los embeddings densos (dense embeddings) representan el significado de un texto en un vector de alta dimensión. Frases semánticamente similares quedan geométricamente cercanas.",
        None,
    ),
    (
        "La búsqueda híbrida combina dense retrieval (semántica) con sparse retrieval (keywords) usando técnicas de fusión como RRF (Reciprocal Rank Fusion) para obtener lo mejor de ambos.",
        None,
    ),
    (
        "FAISS (Facebook AI Similarity Search) es una biblioteca para búsqueda eficiente de vecinos más cercanos en espacios vectoriales. Soporta índices planos, IVF y HNSW.",
        "juan",
    ),
    (
        "Qdrant es un motor de búsqueda vectorial escrito en Rust. Soporta vectores densos, sparse e imágenes, y permite combinar búsqueda vectorial con filtros de metadatos.",
        None,
    ),
    # MapReduce y Spark
    (
        "El paradigma MapReduce procesa datos en dos fases: Map transforma los registros en pares clave-valor, y Reduce agrega todos los valores con la misma clave.",
        None,
    ),
    (
        "Apache Spark introduce transformaciones perezosas (lazy evaluation): no ejecuta ningún cómputo hasta que se llama a una acción como collect(), count() o save().",
        "agustin",
    ),
    (
        "El lineage de un RDD permite reconstruir particiones perdidas sin replicación completa de datos. Spark solo necesita guardar el grafo de transformaciones aplicadas.",
        None,
    ),
    (
        "DataFrames de Spark son una abstracción sobre RDDs con schema explícito. El Catalyst optimizer puede optimizarlos usando reglas algebraicas y estadísticas de columna.",
        None,
    ),
    (
        "Spark MLlib provee algoritmos de ML distribuido: regresión, clasificación, clustering y factorización de matrices. Todos están diseñados para operar sobre RDDs o DataFrames.",
        "agustin",
    ),
    # Probabilístico y aproximado
    (
        "El muestreo estratificado divide la población en grupos homogéneos y muestrea de cada uno proporcionalmente. Garantiza representación de subgrupos raros en el análisis.",
        None,
    ),
    (
        "Los algoritmos aproximados sacrifican exactitud por velocidad o espacio. Son imprescindibles en big data donde el cómputo exacto es prohibitivo en tiempo o memoria.",
        "juan",
    ),
    (
        "MinHash permite estimar la similitud de Jaccard entre conjuntos en O(1) espacio. Es la base del algoritmo LSH (Locality Sensitive Hashing) para deduplicación masiva.",
        None,
    ),
    (
        "Locality Sensitive Hashing (LSH) agrupa en el mismo bucket vectores similares con alta probabilidad. Permite búsqueda de vecinos aproximados sin calcular todas las distancias.",
        None,
    ),
    # Gestión de datos y calidad
    (
        "El linaje de datos (data lineage) rastrea el origen y las transformaciones aplicadas a cada dataset. Es crítico para auditoría, debugging y gobernanza de datos.",
        "juan",
    ),
    (
        "El esquema de evolución permite que los datos almacenados cambien su estructura con el tiempo sin romper los consumidores existentes. Avro y Protobuf lo soportan nativamente.",
        None,
    ),
    (
        "La deduplicación de datos elimina registros duplicados en un dataset. En big data, técnicas como Bloom filters o MinHash permiten detectar duplicados sin cargar todo en memoria.",
        None,
    ),
    (
        "El particionamiento por fecha es un patrón común en data lakes. Almacenar partitions como /año=2026/mes=05/ permite que las consultas con filtro de fecha lean solo las particiones relevantes.",
        "agustin",
    ),
    # Sistemas distribuidos
    (
        "El protocolo Paxos garantiza consenso en sistemas distribuidos tolerantes a fallos de hasta la mitad de los nodos. Es la base de sistemas como ZooKeeper y etcd.",
        None,
    ),
    (
        "Raft es un algoritmo de consenso diseñado para ser más comprensible que Paxos. Separa claramente la elección de líder, la replicación de log y la compactación del estado.",
        None,
    ),
    (
        "El vector clock permite determinar la causalidad entre eventos en sistemas distribuidos sin un reloj global. Dos eventos son concurrentes si ninguno ocurrió antes que el otro.",
        "juan",
    ),
    (
        "La replicación síncrona garantiza durabilidad pero aumenta la latencia de escritura. La replicación asíncrona es más rápida pero puede perder datos en caso de fallo del líder.",
        None,
    ),
    # Búsqueda e IA
    (
        "RAG (Retrieval-Augmented Generation) combina un retriever vectorial con un LLM. El retriever encuentra contexto relevante; el LLM genera la respuesta fundamentada en ese contexto.",
        None,
    ),
    (
        "Los modelos de embeddings multimodales proyectan texto e imágenes al mismo espacio vectorial. Esto permite buscar imágenes con texto o texto con imágenes de forma unificada.",
        "agustin",
    ),
    (
        "El fine-tuning de embeddings adapta un modelo preentrenado a un dominio específico. Mejora la calidad de la recuperación en dominios técnicos o con vocabulario especializado.",
        None,
    ),
    (
        "Cross-encoders re-rankean los candidatos del retriever leyendo query y documento conjuntamente. Son más precisos que bi-encoders pero demasiado lentos para buscar en toda la colección.",
        "juan",
    ),
    (
        "El modelo all-MiniLM-L6-v2 es un bi-encoder de 384 dimensiones optimizado para velocidad en CPU. Genera embeddings de alta calidad para similitud semántica de oraciones.",
        None,
    ),
]

FILLER_NODES: list[dict[str, Any]] = [
    {
        "id": 200 + i,
        "text": text,
        "payload": {"type": "filler", "class": cls},
    }
    for i, (text, cls) in enumerate(_FILLER_TEXTS)
]

# ─── Generador de nodos Wikipedia ────────────────────────────────────────────
# Llama a la API REST de Wikipedia (es) para obtener resúmenes de artículos
# aleatorios. Los resultados se cachean en un fichero JSON local para que
# ejecuciones posteriores sean completamente reproducibles con la misma seed.
#
# IDs sintéticos parten en 1000 para no colisionar con:
#   tesoro (1-11) | ruido (100-157) | relleno (200-258)

_WIKI_API = "https://es.wikipedia.org/api/rest_v1/page/random/summary"
_WIKI_CACHE = Path(__file__).parent / "wiki_cache.json"
_WIKI_USER_AGENT = "DatosMasivos2026-ingest/1.0 (agustin.urrutia@imfd.cl)"
_SYNTHETIC_ID_START = 1000


def _fetch_wiki_articles(n: int, delay: float = 1.0) -> list[dict[str, str]]:
    """
    Llama *n* veces a la API de Wikipedia y devuelve una lista de
    {'title': ..., 'extract': ...}.  Los artículos sin extract se descartan
    y se reintenta hasta conseguir *n* válidos.

    Manejo de rate-limit (HTTP 429): espera con backoff exponencial
    (2 s → 4 s → 8 s …) antes de reintentar.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": _WIKI_USER_AGENT})

    results: list[dict[str, str]] = []
    backoff = 2.0  # segundos de espera inicial ante 429
    print(f"  [wiki] Obteniendo {n} artículos aleatorios de Wikipedia …")

    while len(results) < n:
        try:
            resp = session.get(_WIKI_API, timeout=10)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", backoff))
                wait = max(retry_after, backoff)
                print(
                    f"\n  [wiki] 429 rate-limit — esperando {wait:.0f} s …", flush=True
                )
                time.sleep(wait)
                backoff = min(backoff * 2, 60.0)  # tope en 60 s
                continue

            resp.raise_for_status()
            backoff = 2.0  # reset backoff tras éxito

            data: dict = resp.json()
            extract: str = data.get("extract", "").strip()
            title: str = data.get("title", "").strip()
            if extract and len(extract) > 40:  # descarta stubs muy cortos
                results.append({"title": title, "extract": extract})
                print(
                    f"  [wiki] {len(results)}/{n} — {title[:55]}",
                    end="\r",
                    flush=True,
                )

        except requests.RequestException as exc:
            print(f"\n  [wiki] error de red: {exc}", file=sys.stderr)

        time.sleep(delay)

    print()  # salto de línea tras el \r
    return results


def _load_wiki_cache(path: Path) -> list[dict[str, str]]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_wiki_cache(path: Path, articles: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)


def generate_random_nodes(
    create_n: int,
    id_start: int = _SYNTHETIC_ID_START,
    seed: int = 42,
    cache_path: Path = _WIKI_CACHE,
    request_delay: float = 0.3,
) -> list[dict[str, Any]]:
    """
    Genera *create_n* nodos a partir de resúmenes de Wikipedia (es).

    Reproducibilidad
    ----------------
    La API devuelve artículos no deterministas, por lo que los resultados se
    guardan en *cache_path* (``wiki_cache.json`` junto al script).  En la
    primera ejecución se descargan los artículos; en las siguientes se usa el
    caché y la ``seed`` garantiza el mismo orden y selección.

    Si el caché contiene menos artículos de los requeridos, se descargan los
    que faltan y se actualiza el fichero.

    Args:
        create_n:       Cantidad de nodos a producir.
        id_start:       Primer ID (los IDs serán id_start, id_start+1, …).
        seed:           Semilla para el shuffle reproducible del caché.
        cache_path:     Ruta al fichero JSON de caché.
        request_delay:  Segundos entre llamadas a la API (cortesía de rate-limit).

    Returns:
        Lista de dicts con claves ``'id'``, ``'text'`` y ``'payload'``.
    """
    # 1. Leer caché existente
    cached = _load_wiki_cache(cache_path)

    # 2. Descargar los que falten
    missing = create_n - len(cached)
    if missing > 0:
        print(f"  [wiki] Caché tiene {len(cached)} artículos; faltan {missing}.")
        new_articles = _fetch_wiki_articles(missing, delay=request_delay)
        cached.extend(new_articles)
        _save_wiki_cache(cache_path, cached)
        print(
            f"  [wiki] Caché actualizado → {len(cached)} artículos en '{cache_path.name}'."
        )
    else:
        print(f"  [wiki] Usando caché existente ({len(cached)} artículos).")

    # 3. Shuffle reproducible y selección de los primeros create_n
    pool = list(cached)  # copia para no mutar el caché
    random.Random(seed).shuffle(pool)
    selected = pool[:create_n]

    # 4. Construir nodos
    nodes: list[dict[str, Any]] = []
    for i, article in enumerate(selected):
        nodes.append(
            {
                "id": id_start + i,
                "text": article["extract"],
                "payload": {
                    "type": "synthetic",
                    "source": "wikipedia_es",
                    "title": article["title"],
                },
            }
        )
    return nodes


# ─── Funciones de embedding ───────────────────────────────────────────────────


def make_dense(model: SentenceTransformer, text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


def make_sparse(model: SparseTextEmbedding, text: str) -> models.SparseVector:
    result = next(iter(model.embed([text])))
    return models.SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


# ─── Colección ────────────────────────────────────────────────────────────────


def create_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"[!] La colección '{COLLECTION}' ya existe → eliminando para recrear.")
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )
    print(
        f"[+] Colección '{COLLECTION}' creada (dense: {DENSE_DIM}d coseno | sparse: BM25)."
    )

    # Índice de payload para el campo 'class' (requerido por filtros keyword)
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="class",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print("[+] Índice de payload creado para campo 'class' (keyword).")


# ─── Subida ───────────────────────────────────────────────────────────────────


def upload(
    client: QdrantClient,
    nodes: list[dict[str, Any]],
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    label: str,
) -> None:
    total = len(nodes)
    points: list[models.PointStruct] = []

    for i, node in enumerate(nodes):
        text = node["text"]
        dense_vec = node.get("vector") or make_dense(dense_model, text)
        sparse_vec = make_sparse(sparse_model, text)

        points.append(
            models.PointStruct(
                id=node["id"],
                vector={"dense": dense_vec, "sparse": sparse_vec},
                payload={"text": text, **node["payload"]},
            )
        )

        if len(points) == BATCH or i == total - 1:
            client.upsert(collection_name=COLLECTION, points=points)
            print(f"  [{label}] {min(i + 1, total)}/{total} subidos", end="\r")
            points = []

    print()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:

    # Conexión
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    try:
        client.get_collections()
    except Exception as e:
        print(f"[✗] No se pudo conectar: {e}", file=sys.stderr)
        sys.exit(1)
    print("[+] Conexión exitosa.")

    # Modelos
    print(f"\n[*] Cargando modelo denso  '{DENSE_MODEL}' …")
    dense_model = SentenceTransformer(DENSE_MODEL)
    print(f"[*] Cargando modelo sparse '{SPARSE_MODEL}' …")
    sparse_model = SparseTextEmbedding(SPARSE_MODEL)
    print("[+] Modelos listos.\n")

    # Colección
    create_collection(client)

    # Nodos
    synthetic_nodes = generate_random_nodes(create_n=200)
    all_groups = [
        (TREASURE_NODES, "Tesoro   "),
        (NOISE_NODES, "Ruido    "),
        (FILLER_NODES, "Relleno  "),
        (synthetic_nodes, "Sintético"),
    ]
    for nodes, label in all_groups:
        print(f"[*] Subiendo {len(nodes):>3} nodos de {label} …")
        upload(client, nodes, dense_model, sparse_model, label)

    # Verificación
    info = client.get_collection(COLLECTION)
    print(f"\n[✓] Ingesta completa. Total en colección: {info.points_count} puntos.")

    # Confirmar que los 11 IDs tesoro existen
    result = client.retrieve(
        collection_name=COLLECTION,
        ids=list(range(1, 12)),
        with_payload=False,
        with_vectors=False,
    )
    found = {r.id for r in result}
    missing = set(range(1, 12)) - found
    if missing:
        print(
            f"[!] ADVERTENCIA — IDs tesoro faltantes: {sorted(missing)}",
            file=sys.stderr,
        )
    else:
        print("[✓] Los 11 nodos tesoro (IDs 1-11) están presentes.")

    print("\n─── IDs de nodos tesoro (para la Guía del Maestro) ───")
    for n in TREASURE_NODES:
        paso = n["payload"].get("step", "easter_egg")
        print(f"  ID={n['id']:>2}  paso={paso}  {n['text'][:70]}…")


if __name__ == "__main__":
    main()
