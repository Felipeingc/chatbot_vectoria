require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const path = require('path');
const { createClient } = require('@supabase/supabase-js'); // Importar cliente Supabase

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '5mb' }));
app.use(express.static(__dirname)); // Sirve archivos estáticos desde el directorio actual

// Configuración de Gemini AI
if (!process.env.GEMINI_API_KEY) {
  console.error("Error crítico: La variable de entorno GEMINI_API_KEY no está definida.");
  process.exit(1);
}
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Configuración del cliente de Supabase
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY; // Esta debería ser tu service_role key

if (!supabaseUrl || !supabaseKey) {
  console.error("Error crítico: SUPABASE_URL o SUPABASE_KEY no están definidas en .env.");
  process.exit(1);
}
const supabase = createClient(supabaseUrl, supabaseKey);

// Función para buscar en Supabase (Adaptada para tu tabla de FAQ)
/**
 * Busca una respuesta en la tabla de FAQ de Supabase.
 * @param {string} userQuestion La pregunta del usuario.
 * @returns {Promise<string|null>} La respuesta encontrada o un mensaje indicando que no se encontró.
 */
async function searchSupabase(userQuestion) {
  // -------------------------------------------------------------------------
  // Nombres de tu tabla y columnas (¡YA ACTUALIZADOS CON TU INFORMACIÓN!)
  // -------------------------------------------------------------------------
  const TU_NOMBRE_DE_TABLA_FAQ = 'informacion';
  const TU_COLUMNA_DE_PREGUNTA = 'pregunta';
  const TU_COLUMNA_DE_RESPUESTA = 'respuesta';
  // -------------------------------------------------------------------------

  if (!userQuestion || userQuestion.trim() === '') {
    return "No se proporcionó una pregunta para buscar.";
  }

  const searchTerm = userQuestion.trim().replace(/'/g, "''"); // Escapar comillas simples para SQL

  try {
    const { data, error } = await supabase
      .from(TU_NOMBRE_DE_TABLA_FAQ)
      .select(`${TU_COLUMNA_DE_PREGUNTA}, ${TU_COLUMNA_DE_RESPUESTA}`)
      .textSearch(TU_COLUMNA_DE_PREGUNTA, `'${searchTerm}'`, {
        // config: 'spanish', // Descomenta y ajusta si necesitas especificar idioma para la búsqueda
        type: 'websearch'
      })
      .limit(1);

    if (error) {
      console.error(`Error buscando en Supabase (tabla: ${TU_NOMBRE_DE_TABLA_FAQ}):`, error.message);
      return `Hubo un error técnico al consultar la base de datos. Detalles: ${error.message}`;
    }

    if (data && data.length > 0) {
      const faqEncontrada = data[0];
      console.log(`Información encontrada para "${userQuestion}" (pregunta original de la DB: "${faqEncontrada[TU_COLUMNA_DE_PREGUNTA]}")`);
      return faqEncontrada[TU_COLUMNA_DE_RESPUESTA];
    } else {
      return `No encontré una respuesta directa para "${userQuestion}" en la base de datos de información.`;
    }
  } catch (err) {
    console.error('Excepción durante la búsqueda en Supabase:', err);
    return `Ocurrió una excepción al intentar consultar la base de datos: ${err.message}`;
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

    // 1. Buscar contexto en Supabase
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

    // 2. Construir el prompt para Gemini con el contexto
    const systemInstruction = `Eres Borges IA, un asistente virtual experto. Tu tarea es responder la pregunta del usuario.
Utiliza la siguiente información extraída de una base de datos para fundamentar tu respuesta.
Si la pregunta puede responderse con la información proporcionada, basa tu respuesta *principalmente* en ella.
Si la información no es suficiente, no es relevante para la pregunta, o si el contexto indica que no se encontró información, puedes usar tu conocimiento general, pero si lo haces, podrías mencionar brevemente que la información específica no estaba en la base de datos o que estás complementando.
Evita inventar información si no la tienes. Si el contexto es un mensaje de error de la base de datos, informa al usuario que hubo un problema técnico al buscar la información.

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
    
    const modelName = "gemini-1.5-flash-latest"; // Puedes cambiarlo al modelo específico que prefieras si es necesario
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