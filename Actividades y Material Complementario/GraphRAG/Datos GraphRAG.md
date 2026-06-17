# GraphRAG

## Datos

<https://drive.google.com/file/d/1R0zuErgua14hCzlRZnwVA0ATZFw6QxJu/view?usp=sharing>

## Modelo

Labels de nodos:

* Parlamentario: camara, final, inicio, parl_id, vigente, nombre_completo, fecha_nacimiento: 1944-03-22Z, camara_actual
* Unidad: nombre
* Partido: nombre
* Embedding: value
* Participacion: camara, fecha, sesion, legislatura, tipo_sesion, tipo_participacion, parlamentarios, texto_principal

Relaciones:

* Parlamentario -[:enPartido]->Partido
* Parlamentario -[:enUnidad]->Unidad
* Parlamentario -[:enParticipacion]->Participacion
* Embedding -[:tieneParticipacion]->Participacion
