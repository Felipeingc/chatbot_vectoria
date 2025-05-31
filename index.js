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

async function searchSupabase(userQuestion) {
    console.log(`[searchSupabase] Recibida pregunta del usuario: "${userQuestion}"`);

    if (!userQuestion || userQuestion.trim() === '') {
        return "No se proporcionó una pregunta para buscar.";
    }

    const embeddingModelName = "text-embedding-004";
    const embeddingGenModel = genAI.getGenerativeModel({ model: embeddingModelName });

    let queryEmbeddingVector;
    try {
        console.log(`[searchSupabase] Generando embedding para la consulta con el modelo: ${embeddingModelName}...`);
        const result = await embeddingGenModel.embedContent({
            content: { parts: [{ text: userQuestion }] }, // <<< --- CAMBIO IMPORTANTE AQUÍ ---
            taskType: "RETRIEVAL_QUERY"
        });
        queryEmbeddingVector = result.embedding.values;
        console.log("[searchSupabase] Embedding para la consulta del usuario generado exitosamente.");
    } catch (e) {
        console.error("[searchSupabase] Error al generar embedding para la consulta del usuario:", e);
        return "Hubo un error al procesar tu pregunta para la búsqueda (falla en la generación del embedding).";
    }

    if (!queryEmbeddingVector) {
        return "No se pudo generar el vector de búsqueda para tu pregunta.";
    }

    const matchThreshold = 0.2;
    const matchCount = 7;

    console.log(`[searchSupabase] Llamando a la función RPC 'match_informacion' con umbral: ${matchThreshold}, cantidad: ${matchCount}`);
    try {
        const { data, error } = await supabase.rpc('match_informacion', {
            query_embedding: queryEmbeddingVector,
            match_threshold: matchThreshold,
            match_count: matchCount
        });

        if (error) {
            console.error('[searchSupabase] Error al llamar a RPC match_informacion:', error);
            return `Error al realizar la búsqueda semántica en la base de datos: ${error.message}`;
        }

        if (data && data.length > 0) {
            console.log(`[searchSupabase] Se encontraron ${data.length} coincidencias desde RPC.`);
            let context = "Basado en la información encontrada en la base de datos con búsqueda semántica:\n";
            data.forEach((item, index) => {
                context += `\nFragmento relevante ${index + 1} (Similitud: ${item.similarity.toFixed(2)}):\n`;
                context += `- Respuesta: ${item.respuesta_original}\n`;
            });
            console.log("[searchSupabase] Contexto preparado para Gemini:", context.substring(0, 500) + "...");
            return context;
        } else {
            console.log('[searchSupabase] No se encontraron coincidencias en la base de datos para la búsqueda semántica.');
            return "Después de una búsqueda semántica, no encontré información directamente relevante para tu pregunta en la base de datos.";
        }
    } catch (e) {
        console.error('[searchSupabase] Excepción al llamar a RPC o procesar resultados:', e);
        return `Ocurrió una excepción durante la búsqueda semántica en la base de datos: ${e.message}`;
    }
}

// Endpoint para el streaming de respuestas del chatbot
app.post('/stream', async (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  try {
    const clientConversationHistory = req.body.history || [];

    if (!clientConversationHistory.length) {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ error: "No se recibió historial de conversación." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    const lastMessage = clientConversationHistory[clientConversationHistory.length - 1];
    const userQuestion = lastMessage.parts[0].text.trim();

    if (!userQuestion) {
      if (!res.writableEnded) {
        res.write(`data: ${JSON.stringify({ texto: "La pregunta no puede estar vacía." })}\n\n`);
        res.write('data: [END]\n\n');
        res.end();
      }
      return;
    }

    let contextFromSupabase = "No se pudo realizar la búsqueda en la base de datos o no se encontró información relevante.";
    try {
      const searchResult = await searchSupabase(userQuestion);
      if (searchResult) {
        contextFromSupabase = searchResult;
      }
    } catch (e) {
      console.error("Error al obtener contexto de Supabase:", e);
      contextFromSupabase = `Error al acceder a la base de datos para obtener contexto: ${e.message}`;
    }

    const systemInstruction = `Eres Borges IA, un asistente virtual experto. Tu tarea es responder la pregunta del usuario.
Utiliza la siguiente información extraída de una base de datos para fundamentar tu respuesta.
Si la pregunta puede responderse con la información proporcionada, basa tu respuesta *principalmente* en ella.
Si la información no es suficiente, no es relevante para la pregunta, o si el contexto indica que no se encontró información (por ejemplo, si el contexto dice "No encontré información..."), puedes usar tu conocimiento general, pero si lo haces, podrías mencionar brevemente que la información específica no estaba en la base de datos o que estás complementando.
Evita inventar información si no la tienes. Si el contexto es un mensaje de error de la base de datos (por ejemplo, si el contexto dice "Error al realizar la búsqueda..."), informa al usuario que hubo un problema técnico al buscar la información y que no puedes acceder a los datos específicos en este momento.

Contexto de la base de datos:
---
${contextFromSupabase}
---
`;

    const modifiedHistoryForGemini = JSON.parse(JSON.stringify(clientConversationHistory));

    if (modifiedHistoryForGemini.length > 0) {
      const lastUserTurn = modifiedHistoryForGemini[modifiedHistoryForGemini.length - 1];
      if (lastUserTurn.role === "user") {
        lastUserTurn.parts[0].text = `${systemInstruction}\nPregunta del usuario: ${userQuestion}`;
      }
    }
    
    const modelName = "gemini-2.5-flash-preview-05-20";
    const model = genAI.getGenerativeModel({ model: modelName });

    const streamingResp = await model.generateContentStream({ contents: modifiedHistoryForGemini });

    let streamClosed = false;
    req.on('close', () => {
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
        console.error("Error extrayendo texto del chunk de Gemini:", e);
        continue;
      }

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

  } catch (error) {
    console.error("Error en el endpoint /stream:", error);
    if (!res.headersSent) {
        res.setHeader('Content-Type', 'text/event-stream');
    }
    if (!res.writableEnded) {
        try {
            res.write(`data: ${JSON.stringify({ error: 'Error del servidor al procesar la solicitud: ' + error.message })}\n\n`);
            res.write('data: [END]\n\n');
        } catch (e) {
            console.error("Error al escribir el mensaje de error en el stream:", e);
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