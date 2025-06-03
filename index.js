require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const path = require('path');
const { createClient } = require('@supabase/supabase-js');

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static(__dirname));

// Configuración de Gemini AI
if (!process.env.GEMINI_API_KEY) {
  console.error("Error crítico: La variable de entorno GEMINI_API_KEY no está definida.");
  process.exit(1);
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Configuración del cliente de Supabase
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error("Error crítico: SUPABASE_URL o SUPABASE_KEY no están definidas en .env.");
  process.exit(1);
}
const supabase = createClient(supabaseUrl, supabaseKey);

// Función RAG (sin cambios respecto a tu última versión funcional)
async function searchSupabase(userQuestion) {
    console.log(`[searchSupabase] Recibida pregunta del usuario: "${userQuestion}"`);

    if (!userQuestion || userQuestion.trim() === '') {
        console.log("[searchSupabase] Pregunta vacía recibida.");
        return "No se proporcionó una pregunta para buscar.";
    }

    const embeddingModelName = "text-embedding-004";
    const embeddingModel = genAI.getGenerativeModel({ model: embeddingModelName });

    let queryEmbeddingVector;
    try {
        console.log(`[searchSupabase] Generando embedding para la consulta con el modelo: ${embeddingModelName} para el texto: "${userQuestion}"`);
        const result = await embeddingModel.embedContent({
            content: { parts: [{ text: userQuestion }] }, 
            taskType: "RETRIEVAL_QUERY"
        });

        if (result && result.embedding && Array.isArray(result.embedding.values)) {
            queryEmbeddingVector = result.embedding.values;
        } else if (result && Array.isArray(result.embedding)) { 
            queryEmbeddingVector = result.embedding;
        }
        
        if (!queryEmbeddingVector || !Array.isArray(queryEmbeddingVector)) {
            console.error("[searchSupabase] El vector de embedding obtenido NO es un array válido. 'result.embedding' fue:", result ? result.embedding : "resultado nulo de Gemini");
            throw new Error("Formato de embedding inesperado o nulo de la API de Gemini.");
        }
        console.log("[searchSupabase] Embedding para la consulta del usuario generado exitosamente (primeros 3):", queryEmbeddingVector.slice(0,3));
    } catch (e) {
        console.error("[searchSupabase] Error al generar embedding para la consulta del usuario:", e); // Imprimimos el error completo
        return `Hubo un error al procesar tu pregunta para la búsqueda (falla en la generación del embedding): ${e.message}`;
    }

    const matchThreshold = 0.2;
    const matchCount = 7; 

    const rpcFunctionName = 'match_chunks_vectoria';

    console.log(`[searchSupabase] Llamando a la función RPC '${rpcFunctionName}' con umbral: ${matchThreshold}, cantidad: ${matchCount}`);
    try {
        const { data, error } = await supabase.rpc(rpcFunctionName, { 
            query_embedding: queryEmbeddingVector,
            match_threshold: matchThreshold,
            match_count: matchCount
        });

        if (error) {
            console.error(`[searchSupabase] Error al llamar a RPC ${rpcFunctionName}:`, error);
            return `Error al realizar la búsqueda semántica en la base de datos: ${error.message}`;
        }

        if (data && data.length > 0) {
            console.log(`[searchSupabase] Se encontraron ${data.length} coincidencias desde RPC.`);
            let context = "Contexto relevante encontrado en la base de datos (documentos de vectorIA):\n";
            
            data.forEach((item, index) => {
                context += `\n--- Fragmento ${index + 1} ---\n`;
                context += `Fuente: '${item.nombre_archivo_original}'`; 
                if (item.pagina_inicio != null && item.pagina_fin != null) {
                    context += `, Página(s): ${item.pagina_inicio}-${item.pagina_fin}`;
                }
                context += ` (Similitud: ${item.similitud != null ? item.similitud.toFixed(3) : 'N/A'})\n`;
                context += `Contenido del fragmento:\n${item.texto_chunk_coincidente}\n`; 
                context += `--- Fin Fragmento ${index + 1} ---\n`;
            });
            return context;
        } else {
            console.log(`[searchSupabase] No se encontraron coincidencias en la base de datos para la búsqueda semántica usando ${rpcFunctionName}.`);
            return "Después de una búsqueda semántica en los documentos de vectorIA, no encontré información directamente relevante para tu pregunta.";
        }
    } catch (e) {
        console.error(`[searchSupabase] Excepción al llamar a RPC ${rpcFunctionName} o procesar resultados:`, e);
        return `Ocurrió una excepción durante la búsqueda semántica en la base de datos: ${e.message}`;
    }
}

// --- INICIO DE NUEVAS FUNCIONES PARA AGENDAMIENTO ---

async function detectarIntencionAgendamiento(textoUsuario, historialConversacion = []) {
    console.log(`[IntentDetection] Detectando intención para: "${textoUsuario}"`);
    try {
        let promptContext = "";
        if (historialConversacion.length > 0) {
            promptContext = "Historial de conversación reciente (el último mensaje es el actual del usuario):\n";
            const ultimosTurnos = historialConversacion.slice(-6); // Últimos 3 intercambios
            ultimosTurnos.forEach(turno => {
                // Asegurarse que turno.parts y turno.parts[0] existen
                const turnText = (turno.parts && turno.parts.length > 0 && turno.parts[0].text) ? turno.parts[0].text : "";
                promptContext += `${turno.role === "user" ? "Usuario" : "Chatbot"}: ${turnText}\n`;
            });
            promptContext += "\n---\n";
        }

        const prompt = `${promptContext}Eres un asistente clasificador de intenciones. Analiza el ÚLTIMO mensaje del usuario (considerando el historial previo si se proporciona). Tu única tarea es determinar si el usuario está expresando la intención de agendar una reunión, coordinar una llamada, pedir una cita, o si está preguntando por disponibilidad para hablar. Responde ÚNICAMENTE con una de estas dos palabras: 'AGENDAR_REUNION' si la intención es clara en el último mensaje, o 'OTRA_COSA' en caso contrario. Último mensaje del usuario (ya incluido en el historial si se proporcionó): "${textoUsuario}"`;
        
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
        const result = await model.generateContent(prompt);
        const response = await result.response;
        const text = response.text().trim().toUpperCase(); // Convertir a mayúsculas para comparación robusta

        console.log(`[IntentDetection] Respuesta de Gemini para intención: "${text}"`);
        if (text === 'AGENDAR_REUNION') {
            return 'AGENDAR_REUNION';
        }
        return 'OTRA_COSA';
    } catch (error) {
        console.error("[IntentDetection] Error al detectar intención:", error);
        return 'OTRA_COSA'; 
    }
}

// REEMPLAZA COMPLETAMENTE tu función manejarFlujoAgendamiento con esta:
async function manejarFlujoAgendamiento(userQuestion, conversationHistory, res) {
    console.log("[SchedulingFlow] Procesando agendamiento. Último mensaje usuario:", userQuestion);
    
    let schedulingConvoHistory = "";
    conversationHistory.forEach(turn => {
        if (turn && turn.parts && turn.parts.length > 0 && turn.parts[0].text) {
            schedulingConvoHistory += `${turn.role === "user" ? "Usuario" : "Chatbot"}: ${turn.parts[0].text}\n`;
        }
    });

    const currentDate = new Date(new Date().toLocaleString("en-US", {timeZone: "America/Santiago"})).toISOString().split('T')[0];

    const promptSlotFilling = `
Eres un asistente experto de VectorIA extrayendo información para agendar reuniones.
Tu ÚNICA tarea es analizar el historial de conversación y el ÚLTIMO MENSAJE DEL USUARIO para extraer los siguientes datos si están presentes:
1. motivo (string conciso sobre el tema de la reunión)
2. fecha (string en formato YYYY-MM-DD, interpretando "hoy", "mañana", "próximo lunes", etc. Fecha actual para referencia: ${currentDate}. Incluye siempre el año.)
3. hora (string en formato HH:MM de 24h, interpretando "AM/PM", "tarde", "mañana", etc. Si es ambiguo como "tarde", devuelve null para la hora.)

Historial de conversación (el último mensaje "Usuario:" es el más reciente y relevante para extraer información):
${schedulingConvoHistory}

Responde ÚNICAMENTE con un objeto JSON que contenga las claves "motivo", "fecha", y "hora".
Si un dato no se puede extraer del último mensaje del usuario o no está presente, su valor en el JSON debe ser null.
Si el usuario cancela o dice que no quiere agendar, responde: {"motivo": "CANCELADO", "fecha": null, "hora": null}.
No añadas explicaciones ni texto adicional fuera del JSON. Solo el JSON.
`;

    let datosAgendamiento = { motivo: null, fecha: null, hora: null };

    // Intentar extraer datos de turnos previos en el historial si los tenemos estructurados
    // Esta parte es más avanzada; por ahora, nos enfocaremos en lo que el LLM extrae del último turno + historial.
    // Para simplificar, asumimos que el LLM en cada paso re-evaluará el historial completo
    // y extraerá todo lo que pueda.

    try {
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });
        const generationConfig = { responseMimeType: "application/json" }; // Pedir JSON directamente
        const result = await model.generateContent({
            contents: [{ role: "user", parts: [{text: promptSlotFilling }] }],
            generationConfig // Añadido para forzar JSON
        });

        const response = await result.response;
        let llmResponseText = response.text().trim();
        console.log("[SchedulingFlow] Respuesta LLM para extracción de slots:", llmResponseText);

        let extractedSlots = {};
        try {
            // El LLM ya debería devolver JSON directo con responseMimeType:"application/json"
            extractedSlots = JSON.parse(llmResponseText);
        } catch (jsonError) {
            console.error("[SchedulingFlow] Error parseando JSON de LLM para slots:", jsonError, "Respuesta original:", llmResponseText);
            if (!res.writableEnded) res.write(`data: ${JSON.stringify({ texto: "Hmm, tuve un pequeño problema procesando los detalles. ¿Podrías intentarlo de nuevo?" })}\n\n`);
            if (!res.writableEnded) res.write('data: [END_OF_SCHEDULE_ERROR]\n\n');
            if (!res.writableEnded) res.end();
            return;
        }
        
        // Actualizar nuestros datos recolectados con lo que el LLM extrajo del último mensaje,
        // priorizando lo nuevo si existe.
        // Para un manejo de estado más robusto, necesitaríamos guardar `datosAgendamiento` entre llamadas.
        // Por ahora, si el LLM es bueno, re-extraerá todo del historial.
        // O, más simple, asumimos que el LLM nos da el estado completo de los slots cada vez.
        datosAgendamiento.motivo = extractedSlots.motivo || datosAgendamiento.motivo; // Conserva el anterior si el nuevo es null
        datosAgendamiento.fecha = extractedSlots.fecha || datosAgendamiento.fecha;
        datosAgendamiento.hora = extractedSlots.hora || datosAgendamiento.hora;

        console.log("[SchedulingFlow] Datos de agendamiento después de extracción/actualización:", datosAgendamiento);

        // Lógica de Backend para decidir qué hacer/preguntar
        let botResponseText = "";
        let endSignal = "[CONTINUE_SCHEDULING_FLOW]";

        if (datosAgendamiento.motivo === "CANCELADO") {
            botResponseText = "Entendido. Si cambias de opinión o necesitas ayuda con otra cosa, no dudes en preguntar.";
            endSignal = "[END_OF_CONVERSATION]";
        } else if (!datosAgendamiento.motivo) {
            // Si el historial es muy corto (ej. primer turno de agendamiento) y el motivo no se infirió, preguntar.
            if (conversationHistory.filter(t => t.role==='model').length < 2) { // Menos de 2 respuestas del bot en el flujo de agendamiento
                 botResponseText = "¡Claro! Para agendar, ¿cuál sería el motivo o tema principal de la reunión?";
            } else { // Si ya estamos en el flujo y el motivo se perdió o no se dio
                 botResponseText = "No he podido determinar el motivo de la reunión. ¿Podrías indicármelo, por favor?";
            }
        } else if (!datosAgendamiento.fecha) {
            botResponseText = `Entendido, la reunión es sobre "${datosAgendamiento.motivo}". ¿Para qué fecha te gustaría? (Por favor, usa formato DD/MM/AAAA o indica día).`;
        } else if (!datosAgendamiento.hora) {
            botResponseText = `Perfecto, para el ${datosAgendamiento.fecha} sobre "${datosAgendamiento.motivo}". ¿A qué hora te vendría bien? (Por favor, usa formato HH:MM de 24h).`;
        } else {
            // ¡Todos los datos recolectados!
            botResponseText = `¡Genial! ¿Confirmamos una reunión sobre "${datosAgendamiento.motivo}" para el ${datosAgendamiento.fecha} a las ${datosAgendamiento.hora}? (Simulación: Evento NO creado en Calendar aún)`;
            // Aquí, en un siguiente turno, si el usuario dice "sí", se procedería.
            // Por ahora, esta es la última respuesta simulada del flujo de agendamiento.
            // Para manejar el "sí" final, necesitaríamos otro estado o una lógica más compleja en el siguiente turno.
            endSignal = "[AWAITING_FINAL_CONFIRMATION]"; // El frontend puede usar esto para saber que se espera un sí/no
        }
        
        // Si el usuario acaba de confirmar después de AWAITING_FINAL_CONFIRMATION
        // Necesitamos una forma de saber que el *turno anterior del bot* fue la pregunta de confirmación.
        // Esto se complica sin un manejo de estado explícito.
        // Por ahora, si el último mensaje del usuario es un "sí" y todos los datos están, damos un mensaje final.
        const ultimoMensajeUsuario = conversationHistory[conversationHistory.length - 1].parts[0].text.toLowerCase();
        if (datosAgendamiento.motivo && datosAgendamiento.fecha && datosAgendamiento.hora && (ultimoMensajeUsuario === "si" || ultimoMensajeUsuario === "sí" || ultimoMensajeUsuario === "confirmo" || ultimoMensajeUsuario === "correcto")) {
            // Verificar si la penúltima entrada del historial (última del bot) fue una pregunta de confirmación.
            // Esto es heurístico y puede fallar. Un estado sería mejor.
            let penultimoMensajeBot = "";
            if (conversationHistory.length > 1) {
                const turnoBotAnterior = conversationHistory.slice().reverse().find(turn => turn.role === 'model');
                if (turnoBotAnterior) penultimoMensajeBot = turnoBotAnterior.parts[0].text;
            }

            if (penultimoMensajeBot.startsWith("¡Genial! ¿Confirmamos una reunión sobre")) {
                 botResponseText = `¡Excelente! He registrado tu solicitud para la reunión sobre "${datosAgendamiento.motivo}" el ${datosAgendamiento.fecha} a las ${datosAgendamiento.hora}. (Simulación: Aún no se crea el evento en Google Calendar). ¿Puedo ayudarte en algo más?`;
                 endSignal = "[END_OF_SCHEDULE_SUCCESS_SIMULATION]";
            }
        }


        if (!res.writableEnded) res.write(`data: ${JSON.stringify({ texto: botResponseText })}\n\n`);
        if (!res.writableEnded) res.write(`data: ${endSignal}\n\n`);
        if (!res.writableEnded) res.end();

    } catch (error) {
        console.error("[SchedulingFlow] Error CRÍTICO en el flujo de agendamiento:", error);
        if (!res.writableEnded) res.write(`data: ${JSON.stringify({ texto: "Lo siento, estoy teniendo problemas internos para procesar tu solicitud de agendamiento ahora mismo." })}\n\n`);
        if (!res.writableEnded) res.write('data: [END_OF_SCHEDULE_ERROR]\n\n');
        if (!res.writableEnded) res.end();
    }
}

// --- FIN DE NUEVAS FUNCIONES PARA AGENDAMIENTO ---


// Endpoint para el streaming de respuestas del chatbot
app.post('/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  try {
    const clientConversationHistoryWithCurrent = req.body.history || []; // Asumimos que el frontend envía el historial INCLUYENDO el último mensaje del usuario

    if (!clientConversationHistoryWithCurrent.length) {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: "No se recibió historial de conversación." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    const lastTurn = clientConversationHistoryWithCurrent[clientConversationHistoryWithCurrent.length - 1];
    if (!lastTurn || lastTurn.role !== 'user' || !lastTurn.parts || !lastTurn.parts[0] || !lastTurn.parts[0].text) {
        if (!res.writableEnded) {
            res.write(`data: ${JSON.stringify({ error: "El último turno del historial es inválido o no es del usuario." })}\n\n`);
            res.write('data: [END]\n\n');
            res.end();
        }
        return;
    }
    const userQuestion = lastTurn.parts[0].text.trim();

    if (!userQuestion) {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ texto: "La pregunta no puede estar vacía." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    // --- LÓGICA DE INTENCIÓN Y FLUJO MODIFICADA ---
    // Pasamos el historial completo (que incluye la pregunta actual) para la detección de intención
    const intencionDetectada = await detectarIntencionAgendamiento(userQuestion, clientConversationHistoryWithCurrent);
    console.log(`[MainStream] Intención detectada para "${userQuestion}": ${intencionDetectada}`);

    if (intencionDetectada === 'AGENDAR_REUNION') {
        // La función manejarFlujoAgendamiento ahora toma el historial completo también,
        // ya que el prompt para slot filling usa el historial.
        // userQuestion es el último mensaje, que ya está en clientConversationHistoryWithCurrent.
        await manejarFlujoAgendamiento(userQuestion, clientConversationHistoryWithCurrent, res);
        // manejarFlujoAgendamiento se encarga de res.end()
    } else {
        // Intención es 'OTRA_COSA', proceder con el flujo RAG normal
        console.log("[MainStream] Procediendo con búsqueda RAG.");
        let contextFromSupabase = "No se pudo realizar la búsqueda en la base de datos o no se encontró información relevante.";
        try {
          const searchResult = await searchSupabase(userQuestion);
          if (searchResult) {
            contextFromSupabase = searchResult;
          }
        } catch (e) {
          console.error("[MainStream] Error al obtener contexto de Supabase para RAG:", e);
          contextFromSupabase = `Error al acceder a la base de datos para obtener contexto RAG: ${e.message}`;
        }

const systemInstruction = `Eres Funes, el asistente virtual de VectorIA. Tu tarea es responder las preguntas del usuario respecto de nuestra empresa VectorIA.
Utiliza la siguiente información extraída de una base de datos para fundamentar tu respuesta.
Si la pregunta puede responderse con la información proporcionada, basa tu respuesta *principalmente* en ella.
Si la información no es suficiente, no es relevante, o el contexto indica que no se encontró información, usa tu conocimiento general, pero aclara si complementas.
Evita inventar información. Si el contexto es un mensaje de error, informa al usuario sobre el problema técnico.

**INSTRUCCIONES CRÍTICAS DE FORMATO Y ESTILO PARA TU RESPUESTA EN STREAMING:**
1.  **DEBES generar tu respuesta directamente en HTML semántico y simple.** Cada fragmento (chunk) que envíes en el stream ya debe ser HTML válido o texto plano que forme parte de una estructura HTML.
2.  **UTILIZA ETIQUETAS HTML APROPIADAS:**
    * Párrafos: \`<p>Texto del párrafo.</p>\`
    * Negritas: \`<strong>texto importante</strong>\` o \`<b>texto importante</b>\`
    * Listas con viñetas: \`<ul><li>Ítem 1</li><li>Ítem 2</li></ul>\`
    * Listas numeradas: \`<ol><li>Paso 1</li><li>Paso 2</li></ol>\`
    * Tablas: Utiliza una estructura HTML de tabla (\`<table>\`, \`<thead>\`, \`<tbody>\`, \`<tr>\`, \`<th>\`, \`<td>\`).
    * Saltos de línea: Usa \`<br>\` solo si es estrictamente necesario y no se puede lograr con párrafos o estructura de lista.
3.  **NO ENVÍES SINTAXIS MARKDOWN (como \`**\`, \`*\`, \`-\`, \`#\`, \`|\` para tablas). Envía el HTML equivalente directamente.**
4.  **MUY IMPORTANTE: NO ENVUELVAS tu respuesta HTML en bloques de código Markdown como \`\`\`html ... \`\`\` o \`\`\` ... \`\`\`. Proporciona el HTML crudo directamente.**
5.  **NO MENCIONES 'fragmentos', 'base de datos', 'contexto proporcionado', ni cómo obtuviste la información.** Integra la información fluidamente.
6.  **HABLA EN PRIMERA PERSONA PLURAL ('nosotros', 'en VectorIA podemos...')** cuando te refieras a VectorIA.
7.  Estructura bien la información y sé conciso, no abrumes con demasiada información. Evita bloques de texto densos sin formato.
8. Solo saluda al inicio de la conversación, no digas 'Hola' en cada respuesta.

Contexto de la base de datos:
---
${contextFromSupabase}
---
`;
        // Para RAG, el historial previo y la pregunta actual con el contexto se combinan.
        // `clientConversationHistoryWithCurrent` ya tiene el último mensaje del usuario.
        // Necesitamos construir el array `contents` para Gemini.
        // El último elemento "user" debe ser el que tenga el systemInstruction + contexto + pregunta.
        
        const historyForRAG = clientConversationHistoryWithCurrent.slice(0, -1); // Todo excepto el último turno del usuario
        
        const contentsForGeminiRAG = [
            ...historyForRAG,
            { role: "user", parts: [{ text: `${systemInstruction}\nPregunta original del usuario: ${userQuestion}` }] }
        ];
        
        const modelName = "gemini-2.0-flash"; // Modelo para las respuestas RAG
        const model = genAI.getGenerativeModel({ model: modelName });

        console.log("[MainStream] Enviando a Gemini para respuesta RAG...");
        const streamingResp = await model.generateContentStream({ contents: contentsForGeminiRAG });

        let streamClosed = false;
        req.on('close', () => {
            console.log("[MainStream] Cliente desconectado, cerrando stream RAG.");
            streamClosed = true;
        });

for await (const chunk of streamingResp.stream) {
              if (streamClosed || res.writableEnded) break;

              let texto = '';
              try {
                if (chunk && typeof chunk.text === 'function') {
                  texto = chunk.text();
                } else if (!texto && chunk && chunk.candidates?.length > 0 &&
                           chunk.candidates[0].content?.parts?.length > 0 &&
                           typeof chunk.candidates[0].content.parts[0].text === 'string') {
                  texto = chunk.candidates[0].content.parts[0].text;
                }
              } catch (e) {
                console.error("Error extrayendo texto del chunk de Gemini RAG:", e);
                continue;
              }

              // <<< --- INICIO DE NUEVA LÓGICA DE LIMPIEZA --- >>>
              if (typeof texto === "string") {
                // Regex para eliminar ```html ... ``` o ``` ... ```
                // \s* permite espacios opcionales después de ```html o ```
                // ([\s\S]*?) captura cualquier caracter (incluyendo nuevas líneas) de forma no codiciosa
                const markdownCodeBlockRegex = /^```(?:html)?\s*([\s\S]*?)\s*```$/;
                const match = texto.match(markdownCodeBlockRegex);
                if (match && match[1]) {
                  console.log("[BackendClean] Bloque Markdown detectado, extrayendo contenido.");
                  texto = match[1].trim(); // Usar solo el contenido de adentro
                }
                
                // A veces el LLM puede enviar solo la apertura o el cierre en un chunk,
                // o puede haber texto antes/después. Esto es más complejo de limpiar chunk a chunk.
                // Una limpieza más simple es quitar las marcas si aparecen al inicio/final del chunk.
                // Esto es menos robusto que el regex de arriba para bloques completos, pero puede ayudar con chunks parciales.
                if (texto.startsWith("```html")) texto = texto.substring(7).trimStart();
                else if (texto.startsWith("```")) texto = texto.substring(3).trimStart();
                if (texto.endsWith("```")) texto = texto.substring(0, texto.length - 3).trimEnd();

              }
              // <<< --- FIN DE NUEVA LÓGICA DE LIMPIEZA --- >>>

              if (typeof texto === "string" && texto.trim() !== "") {
                if (!streamClosed && !res.writableEnded) {
                    res.write(`data: ${JSON.stringify({ texto })}\n\n`);
                }
              }
            }

        if (!streamClosed && !res.writableEnded) {
            res.write('data: [END]\n\n');
            res.end();
        }
    }

  } catch (error) {
    console.error("Error GRAVE en el endpoint /stream:", error);
    if (!res.headersSent) {
        res.setHeader('Content-Type', 'text/event-stream');
    }
    if (!res.writableEnded) {
        try {
            res.write(`data: ${JSON.stringify({ error: 'Error catastrófico del servidor: ' + error.message })}\n\n`);
            res.write('data: [END_OF_CRITICAL_ERROR]\n\n');
        } catch (e) {
            console.error("Error al escribir el mensaje de error crítico en el stream:", e);
        } finally {
            if (!res.writableEnded) {
                res.end();
            }
        }
    }
  }
});

// Ruta para servir el archivo HTML principal
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'index.html'));
});

// Inicia el servidor
app.listen(port, () => {
  console.log(`Servidor escuchando en http://localhost:${port}`);
  console.log('Asegúrese de que GEMINI_API_KEY, SUPABASE_URL, y SUPABASE_KEY están configuradas en su archivo .env');
});